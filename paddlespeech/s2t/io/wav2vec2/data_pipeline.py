"""A pipeline for data transformations.

Example
-------
>>> from hyperpyyaml import load_hyperpyyaml
>>> yamlstring = '''
... pipeline: !new:speechbrain.utils.data_pipeline.DataPipeline
...     static_data_keys: [a, b]
...     dynamic_items:
...         -   func: !name:operator.add
...             takes: ["a", "b"]
...             provides: foo
...         -   func: !name:operator.sub
...             takes: ["foo", "b"]
...             provides: bar
...     output_keys: ["foo", "bar"]
... '''
>>> hparams = load_hyperpyyaml(yamlstring)
>>> hparams["pipeline"]({"a":1, "b":2})
{'foo': 3, 'bar': 1}

Author:
    * Aku Rouhe
"""

import inspect
from dataclasses import dataclass
from paddlespeech.s2t.io.wav2vec2.depgraph import DependencyGraph

@dataclass
class StaticItem:
    """Data class that represents a static item.

    Static items are in-memory items so they don't need to be computed
    dynamically.
    """

    key: str


class DynamicItem:
    """Essentially represents a data transformation function.

    A DynamicItem takes some arguments and computes its value dynamically when
    called. A straight-forward use-case is to load something from disk
    dynamically; take the path and provide the loaded data.

    Instances of this class are often created implicitly via the
    @takes and @provides decorators or otherwise from specifying the taken and
    provided arguments and the function.

    A counterpart is the GeneratorDynamicItem, which should be used for
    generator functions.

    Arguments
    ---------
    takes : list
        The keys of the items that this needs to compute its output.
    func : callable
        The function that is used to compute the output.
    provides : list
        The keys that this provides.
    """

    def __init__(self, takes=[], func=None, provides=[]):
        self.takes = takes
        self.func = func
        self.provides = provides

    def __call__(self, *args):
        return self.func(*args)

    # The next methods are more about supporting GeneratorDynamicItems
    def next_takes(self):
        """The next argkeys to provide to this, when called."""
        # Regular function DynamicItems always just need the same set of args
        return self.takes

    def next_provides(self):
        """The next keys that this provides, when called."""
        # Regular function DynamicItems always just provide the same set of keys
        return self.provides

    def provided_in_order(self):
        """Assuming that this may need to be called multiple times; which keys
        does it provide at that call. Returns a list, with len equal to the
        number of times that this may be called."""
        # Regular function DynamicItems are only called once:
        return [self.provides]

    def reset(self):
        """Signals that this will not be called any more times on this pipeline
        call."""
        # Regular function DynamicItems don't need special resets.
        pass


class GeneratorDynamicItem(DynamicItem):
    """Essentially represents a multi-step data transformation.

    This is the generator function counterpart for DynamicItem (which should be
    used for regular functions).

    A GeneratorDynamicItem first takes some arguments and then uses those in
    multiple steps to incrementally compute some values when called.

    A typical use-case is a pipeline of transformations on data: e.g. taking in
    text as a string, and first a tokenized version, and then on the second
    call providing an integer-encoded version. This can be used even though the
    integer-encoder needs to be trained on the first outputs.

    The main benefit is to be able to define the pipeline in a clear function,
    even if parts of the pipeline depend on others for their initialization.

    Example
    -------
    >>> lab2ind = {}
    >>> def text_pipeline(text):
    ...     text = text.lower().strip()
    ...     text = "".join(c for c in text if c.isalpha() or c == " ")
    ...     words = text.split()
    ...     yield words
    ...     encoded = [lab2ind[word] for word in words]
    ...     yield encoded
    >>> item = GeneratorDynamicItem(
    ...         func=text_pipeline,
    ...         takes=["text"],
    ...         provides=["words", "words_encoded"])
    >>> # First create the integer-encoding:
    >>> ind = 1
    >>> for token in item("Is this it? - This is it."):
    ...     if token not in lab2ind:
    ...         lab2ind[token] = ind
    ...         ind += 1
    >>> # Now the integers can be encoded!
    >>> item()
    [1, 2, 3, 2, 1, 3]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Doesn't generate electricity, only stores the currently active
        # generator:
        self.current_generator = None
        self.num_provided_items = 0

    def __call__(self, *args):
        if self.num_provided_items == len(self.provides):
            raise RuntimeError("DynamicItemPipeline called too many times!")
        if not self.current_generator:
            self.current_generator = self.func(*args)
        # NOTE: Not supporting sending new values to the pipeline.
        out = next(self.current_generator)
        self.num_provided_items += 1
        return out

    def next_takes(self):
        """The next argkeys to provide to this, when called."""
        if not self.current_generator:
            return self.takes
        else:
            return []

    def next_provides(self):
        """The next keys that this provides, when called."""
        keys = self.provides[self.num_provided_items]
        # Support multiple yielded values like:
        # @yields("wav_read", ["left_ch", "right_ch"])
        if isinstance(keys, str):
            return [keys]
        else:
            return keys

    def provided_in_order(self):
        """Assuming that this may need to be called multiple times; which keys
        does it provide at that call. Returns a list, with len equal to the
        number of times that this may be called."""
        in_order = []
        for keys in self.provides:
            # Support multiple yielded values like:
            # @provides("wav_read", ["left_ch", "right_ch"])
            if isinstance(keys, str):
                in_order.append([keys])
            else:
                in_order.append(keys)
        return in_order

    def reset(self):
        """Signals that this will not be called any more times on this pipeline
        call."""
        if self.current_generator is not None:
            self.current_generator.close()
        self.current_generator = None
        self.num_provided_items = 0


def takes(*argkeys):
    """Decorator which makes a DynamicItem and specifies its argkeys.

    If the wrapped object is a generator function (has a yield statement),
    Creates a GeneratorDynamicItem. If the object is already a DynamicItem,
    just specifies the argkeys for that. Otherwise creates a new regular
    DynamicItem, with argkeys specified.

    The args are always passed to the function at the start. Generators could
    support sending new arguments, but for such use cases, simply create a new
    dynamic item. The GeneratorDynamicItem class is meant for pipelines which
    take in an input and transform it in multiple ways, where the intermediate
    representations may be needed for e.g. fitting a BPE segmenter.

    Example
    -------
    >>> @takes("text")
    ... def tokenize(text):
    ...     return text.strip().lower().split()
    >>> tokenize.provides = ["tokenized"]
    >>> tokenize('\tThis Example gets tokenized')
    ['this', 'example', 'gets', 'tokenized']
    """

    def decorator(obj):
        """Decorator definition."""
        if isinstance(obj, DynamicItem):
            if obj.takes:
                raise ValueError("Can't overwrite DynamicItem.takes")
            obj.takes = argkeys
            return obj
        elif inspect.isgeneratorfunction(obj):
            return GeneratorDynamicItem(takes=argkeys, func=obj)
        else:
            return DynamicItem(takes=argkeys, func=obj)

    return decorator

takes_decorator = takes  # Just for DataPipeline.add_dynamic_item

def provides(*output_keys):
    """Decorator which makes a DynamicItem and specifies what keys it provides.

    If the wrapped object is a generator function (has a yield statement),
    Creates a GeneratorDynamicItem. If the object is already a DynamicItem,
    just specifies the provided keys for that. Otherwise creates a new regular
    DynamicItem, with provided keys specified.

    NOTE
    ----
    The behavior is slightly different for generators and regular functions, if
    many output keys are specified, e.g. @provides("signal", "mfcc"). Regular
    functions should return a tuple with len equal to len(output_keys), while
    generators should yield the items one by one.

    >>> @provides("signal", "feat")
    ... def read_feat():
    ...     wav = [.1,.2,-.1]
    ...     feat = [s**2 for s in wav]
    ...     return wav, feat
    >>> @provides("signal", "feat")
    ... def read_feat():
    ...     wav = [.1,.2,-.1]
    ...     yield wav
    ...     feat = [s**2 for s in wav]
    ...     yield feat

    If multiple keys are yielded at once, write e.g.,

    >>> @provides("wav_read", ["left_channel", "right_channel"])
    ... def read_multi_channel():
    ...     wav = [[.1,.2,-.1],[.2,.1,-.1]]
    ...     yield wav
    ...     yield wav[0], wav[1]

    """

    def decorator(obj):
        """Decorator definition."""
        if isinstance(obj, DynamicItem):
            if obj.provides:
                raise ValueError("Can't overwrite DynamicItem provides-list.")
            obj.provides = output_keys
            return obj
        elif inspect.isgeneratorfunction(obj):
            return GeneratorDynamicItem(func=obj, provides=output_keys)
        else:
            return DynamicItem(func=obj, provides=output_keys)

    return decorator


provides_decorator = provides  # Just for DataPipeline.add_dynamic_item


class DataPipeline:
    """Organises data transformations into a pipeline.

    Example
    -------
    >>> pipeline = DataPipeline(
    ...     static_data_keys=["text"],
    ...     dynamic_items=[
    ...     {"func": lambda x: x.lower(), "takes": "text", "provides": "foo"},
    ...     {"func": lambda x: x[::-1], "takes": "foo", "provides": "bar"},
    ...     ],
    ...     output_keys=["bar"],
    ... )
    >>> pipeline({"text": "Test"})
    {'bar': 'tset'}
    """

    def __init__(self, static_data_keys, dynamic_items=[], output_keys=[]):
        self.dg = DependencyGraph()
        self._exec_order = None
        self.key_to_node = {}
        self.unaccounted_keys = {}
        self.dynamic_items = []
        self.output_mapping = {}
        self.add_static_keys(static_data_keys)
        self.add_dynamic_items(dynamic_items)
        self.set_output_keys(output_keys)

    def add_static_keys(self, static_keys):
        """Informs the pipeline about static items.

        Static items are the ones provided to __call__ as data.
        """
        for key in static_keys:
            node_id = self.dg.add_node(data=StaticItem(key=key))
            self.key_to_node[key] = node_id

    def add_dynamic_items(self, dynamic_items):
        """Add multiple dynamic items at once."""
        for item in dynamic_items:
            try:
                self.add_dynamic_item(**item)
            except TypeError:
                self.add_dynamic_item(item)

    def add_dynamic_item(self, func, takes=None, provides=None):
        """Adds a dynamic item to the Pipeline.

        Two calling conventions. For DynamicItem objects, just use:
        add_dynamic_item(dynamic_item)
        But otherwise, should use:
        add_dynamic_item(func, takes, provides)

        Arguments
        ---------
        func : callable, DynamicItem
            If a DynamicItem is given, adds that directly. Otherwise a
            DynamicItem is created, and this specifies the callable to use. If
            a generator function is given, then create a GeneratorDynamicItem.
            Otherwise creates a normal DynamicItem.
        takes : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single key can be given as a bare string.
        provides : str, list
            For regular functions, the key or list of keys that it provides.
            If you give a generator function, key or list of keys that it
            yields, in order. Also see the provides decorator.
            A single key can be given as a bare string.
        """
        if isinstance(func, DynamicItem):
            if takes is not None or provides is not None:
                raise ValueError(
                    "If providing a DynamicItem directly, don't "
                    "specify takes or provides"
                )
            else:
                self._add_dynamic_item_object(func)
                return
        if isinstance(takes, str):
            takes = [takes]
        if isinstance(provides, str):
            provides = [provides]
        di = takes_decorator(*takes)(provides_decorator(*provides)(func))
        self._add_dynamic_item_object(di)

    def _add_dynamic_item_object(self, obj):
        """Internally adds the object.

        There is a node in the dependency graph for each call of the
        DynamicItem. Each call may return multiple keys and depend on multiple
        keys. An internal dict maps key to the id of the node that produces it.
        """
        if not obj.provides:
            raise ValueError(
                "Won't add redundant dynamic item which doesn't "
                "provide anything."
            )
        depended = []
        for key in obj.takes:
            # Might not be accounted for, yet:
            if key not in self.key_to_node:
                dependee_keys = self.unaccounted_keys.setdefault(key, [])
                dependee_keys.extend(obj.next_provides())
            else:
                depended.append(self.key_to_node[key])
        for provided in obj.provided_in_order():
            node_id = self.dg.add_node(data=obj)
            for key in provided:
                self.key_to_node[key] = node_id
                # This key may also be unaccounted for, so account for it now:
                if key in self.unaccounted_keys:
                    for dependee_key in self.unaccounted_keys[key]:
                        dependee_node = self.key_to_node[dependee_key]
                        self.dg.add_edge(dependee_node, node_id)
                    del self.unaccounted_keys[key]  # Now accounted for!
            for dep_id in depended:
                self.dg.add_edge(node_id, dep_id)
            # Next call will depend on this call:
            depended = [node_id]
        # Keep a reference to the item in this object, as well:
        self.dynamic_items.append(obj)

    def set_output_keys(self, keys):
        """Use this to change the output keys.

        Also re-evaluates execution order.
        So if you request different outputs, some parts of the
        data pipeline may be skipped.

        Arguments
        ---------
        keys : dict, list, None
            List of keys (str) to produce in output.

            If a dict is given; it is used to map internal keys to output keys.
            From the output_keys dict key:value pairs the key appears outside,
            and value is the internal key.
        """
        self.output_mapping = self._output_keys_to_mapping(keys)
        self._exec_order = None

    @staticmethod
    def _output_keys_to_mapping(keys):
        # Ensure a mapping (accept a list for convenience, too)
        if keys is None:
            output_mapping = {}
        elif isinstance(keys, dict):
            output_mapping = keys
        else:
            output_mapping = {key: key for key in keys}
        return output_mapping

    def compute_outputs(self, data):
        """
        Arguments
        ---------
        data : dict
            Dictionary with data entries by key.

        Returns
        -------
        dict
            With the keys that were set.
        """
        if self._exec_order is None:
            self._prepare_run(data)
        return self._compute(data, self._exec_order, self.output_mapping)

    def compute_specific(self, keys, data):
        """Compute output of specific item, without changing output_keys."""
        output_mapping = self._output_keys_to_mapping(keys)
        order = self.dg.get_evaluation_order(
            selected_keys=self.get_selected_node_ids(keys)
        )
        return self._compute(data, order, output_mapping)

    def _compute(self, data, order, output_mapping):
        if self.unaccounted_keys:
            MSG = "These keys are still unaccounted for in the data pipeline: "
            MSG += ", ".join(self.unaccounted_keys)
            raise RuntimeError(MSG)
        intermediate = {}
        for node_id, edges, item in order:
            if isinstance(item, StaticItem):
                # Static item in data.
                # Just check that key is found.
                try:
                    data[item.key]
                    continue
                except KeyError:
                    raise KeyError(f"Expected key {item.key} in data!")
            # A dynamic item, which we should compute:
            args = [
                data[argkey] if argkey in data else intermediate[argkey]
                for argkey in item.next_takes()
            ]
            # This needs to be called BEFORE the dynamic item is called.
            provided_keys = item.next_provides()
            values = item(*args)  # Call the DynamicItem to produce output
            # If there is just one output value, wrap in a list so that
            # it can be zipped as well:
            if len(provided_keys) == 1:
                values = [values]
            intermediate.update(zip(provided_keys, values))
        for dynamic_item in self.dynamic_items:
            dynamic_item.reset()
        return {
            outkey: data[inkey] if inkey in data else intermediate[inkey]
            for outkey, inkey in output_mapping.items()
        }

    def get_selected_node_ids(self, selected_keys):
        """Translates selected keys to dependency graph keys."""
        return [self.key_to_node[key] for key in selected_keys]

    def __call__(self, data):
        return self.compute_outputs(data)

    def _prepare_run(self, data):
        self._exec_order = list(
            self.dg.get_evaluation_order(
                self.get_selected_node_ids(self.output_mapping.values())
            )
        )

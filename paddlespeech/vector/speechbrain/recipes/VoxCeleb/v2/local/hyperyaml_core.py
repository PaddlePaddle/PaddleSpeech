"""This library gathers utilities for hyperpyyaml loading

Authors
 * Peter Plantinga 2020
 * Aku Rouhe 2020
"""

import re
import ast
import yaml
import copy
import pydoc
import os.path
import inspect
import functools
import collections
import ruamel.yaml
import operator as op
from io import StringIO


# NOTE: Empty dict as default parameter is fine here since overrides are never
# modified
def load_hyperpyyaml(
    yaml_stream, overrides=None, overrides_must_match=True,
):
    r'''This function implements the HyperPyYAML syntax

    The purpose for this syntax is a compact, structured hyperparameter and
    function definition. This function implements a few extensions to the yaml
    syntax, listed below.

    **PyYAML complex tag shortcuts**

    Part of our clean structured hyperparameter interface is being able to
    specify python objects easily and cleanly. This is possible with
    native YAML using the following syntax:

    .. code-block:: yaml

        alignment_saver: !!python/object/new:speechbrain.data_io.TensorSaver
            kwargs: {save_dir: results/asr/ali}

    However, due to the extensive use within speechbrain yaml files, we have
    added a shortcut for this that has the following syntax:

    .. code-block:: yaml

        alignment_saver: !new:speechbrain.data_io.TensorSaver
            save_dir: results/asr/ali

    In this example, the alignment_saver will be an instance of the
    ``TensorSaver`` class, with ``'exp/asr/ali'`` passed to the
    ``__init__()`` method as a keyword argument. This is equivalent to:

    .. code-block:: python

        import speechbrain.data_io.data_io
        alignment_saver = speechbrain.data_io.TensorSaver(
            save_dir='exp/asr/ali'
        )

    We have also implemented a few more shortcuts:::

        !!python/name: => !name:
        !!python/module: => !module:
        !!python/object/apply: => !apply:

    **References and copies**

    Allows internal references to any node in the file. Any node with
    tag ``!ref`` will create an object reference to the yaml object at the
    ``<key.subkey>`` location within the yaml itself,
    following reference chains.

    .. code-block:: yaml

        output_folder: results/asr
        alignment_saver: !new:speechbrain.data_io.TensorSaver
            save_dir: !ref <output_folder>

    Strings values are handled specially: references are substituted but
    the rest of the string is left in place, allowing filepaths to be
    easily extended:

    .. code-block:: yaml

        output_folder: results/asr
        alignment_saver: !new:speechbrain.data_io.TensorSaver
            save_dir: !ref <output_folder>/ali  # results/asr/ali

    A more complex example for demonstration purposes:

    .. code-block:: yaml

        key1: {a: !new:object {arg1: 1}}
        key2: !ref <key1[a]>

    Here, ``key2`` will contain a reference to the ``a`` object, so changing
    ``a.arg1`` will also change ``key2.arg1``. If you need a
    deep copy of the object instead of a shallow reference, you
    can use a similar syntax with the tag ``!copy``. For example:

    .. code-block:: yaml

        key1: {a: !new:object {arg1: 1}}
        key2: !copy <key1[a]>

    These will also implement very basic arithmetic, so:

    .. code-block:: yaml

        key1: 1
        key2: !ref <key1> + 3  # this is 4

    **Tuples**

    One last minor enhancement is an implicit tuple resolver. Passing
    a string value of ``(3, 4)`` will be given a tag of ``!tuple`` which is
    then interpreted as a tuple.

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string from which to read.
    overrides : mapping or str
        A set of overrides for the values read from the stream.
        As yaml implements a nested structure, so can the overrides.
        See `speechbrain.utils.data_utils.recursive_update`
    overrides_must_match : bool
        Whether an error will be thrown when an override does not match
        a corresponding key in the yaml_stream.
    return_dict : bool
        Whether to return a dictionary rather than the default namespace.

    Returns
    -------
    hparams : dict
        Reflects the structure of ``yaml_stream``.

    Example
    -------
    >>> yaml_string = """
    ... a: 3
    ... thing: !new:collections.Counter
    ...     b: !ref <a>
    ... """
    >>> params = load_hyperpyyaml(yaml_string)
    >>> params["thing"]
    Counter({'b': 3})
    '''
    yaml_stream = resolve_references(
        yaml_stream, overrides, overrides_must_match
    )
    print("step1 yaml: {}".format(yaml_stream))
    # Parse flat tuples (no nesting of lists, dicts)
    yaml.Loader.add_constructor(tag="!tuple", constructor=_make_tuple)
    tuple_pattern = re.compile(r"^\(.*\)$")
    yaml.Loader.add_implicit_resolver("!tuple", tuple_pattern, first="(")

    # Parse shortcuts to `new`, `name`, and `module`
    yaml.Loader.add_multi_constructor("!new:", _construct_object)
    yaml.Loader.add_multi_constructor("!name:", _construct_name)
    yaml.Loader.add_multi_constructor("!module:", _construct_module)
    yaml.Loader.add_multi_constructor("!apply:", _apply_function)

    # NOTE: Here we apply a somewhat dirty trick.
    # We change the yaml object construction to be deep=True by default.
    #
    # Sometimes in e.g. !apply: calls which would get passed a dictionary
    # by reference, the dict got passed empty.
    # This is to do with pyyaml constructors for e.g. mappings and their
    # behaviour with the default deep=False
    # See for example https://stackoverflow.com/a/43812995
    #
    # In our tests nothing seems to break after changing the default.
    # But if later weird things start happening in YAML loading,
    # see this.
    yaml.constructor.BaseConstructor.construct_object.__defaults__ = (
        True,
    )  # deep=True
    hparams = yaml.load(yaml_stream, Loader=yaml.Loader)
    # Change back to normal default:
    yaml.constructor.BaseConstructor.construct_object.__defaults__ = (
        False,
    )  # deep=False

    # Remove items that start with "__"
    removal_keys = [k for k in hparams.keys() if k.startswith("__")]
    for key in removal_keys:
        del hparams[key]

    return hparams


class RefTag:
    """Class for dumping !ref tags to yaml

    Arguments
    ---------
    ref_str : str
        String including yaml keys in `<key>` notation

    Example
    -------
    See ``dump_hyperpyyaml``
    """

    yaml_tag = "!ref"

    def __init__(self, ref_str):
        self.ref_str = ref_str

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag, node.ref_str)


class Placeholder:
    """Class for dumping !PLACEHOLDER tags to yaml

    Example
    -------
    See ``dump_hyperpyyaml``
    """

    yaml_tag = "!PLACEHOLDER"

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag, "")


def dump_hyperpyyaml(yaml_tree, output_stream, *args, **kwargs):
    r"""Dump yaml including placeholder and reference tags.

    Arguments
    ---------
    yaml_tree : dict
        An object to dump
    output_stream : stream
        A file stream for putting the yaml
    *args, **kwargs
        Arguments to forward to ruamel.yaml.YAML().dump()

    Example
    -------
    >>> to_yaml = {'a': Placeholder(), 'b': RefTag('<a>')}
    >>> stringio = StringIO()
    >>> dump_hyperpyyaml(to_yaml, stringio)
    >>> stringio.getvalue()
    'a: !PLACEHOLDER\nb: !ref <a>\n'
    """
    ruamel_yaml = ruamel.yaml.YAML()
    ruamel_yaml.representer.add_representer(RefTag, RefTag.to_yaml)
    ruamel_yaml.representer.add_representer(Placeholder, Placeholder.to_yaml)
    ruamel_yaml.dump(yaml_tree, output_stream, *args, **kwargs)


def resolve_references(yaml_stream, overrides=None, overrides_must_match=False):
    r'''Resolves inter-document references, a component of HyperPyYAML.

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string with the contents of a yaml file
        written with the HyperPyYAML syntax.
    overrides : mapping or str
        Replacement values, either in a yaml-formatted string or a dict.
    overrides_must_match : bool
        Whether an error will be thrown when an override does not match
        a corresponding key in the yaml_stream. This is the opposite
        default from ``load_hyperpyyaml`` because ``resolve_references``
        doesn't need to be as strict by default.

    Returns
    -------
    stream
        A yaml-formatted stream with all references and overrides resolved.

    Example
    -------
    >>> yaml_string = """
    ... constants:
    ...     a: 3
    ...     b: !ref <constants[a]>
    ... """
    >>> overrides = {'constants': {'a': 4}}
    >>> resolve_references(yaml_string, overrides).getvalue()
    'constants:\n  a: 4\n  b: 4\n'
    '''
    # find imported yaml location relative to main yaml file
    file_path = None
    print(yaml_stream)
    if hasattr(yaml_stream, "name"):
        file_path = os.path.dirname(os.path.realpath(yaml_stream.name))

    # Load once to store references and apply overrides
    # using ruamel.yaml to preserve the tags
    ruamel_yaml = ruamel.yaml.YAML()
    preview = ruamel_yaml.load(yaml_stream)
    print("preview: {}".format(preview))
    if overrides is not None and overrides != "":
        if isinstance(overrides, str):
            overrides = ruamel_yaml.load(overrides)
        recursive_update(preview, overrides, must_match=overrides_must_match)
    # 处理copy和ref标签
    _walk_tree_and_resolve("root", preview, preview, overrides, file_path)

    # Dump back to string so we can load with bells and whistles
    yaml_stream = StringIO()
    ruamel_yaml.dump(preview, yaml_stream)
    yaml_stream.seek(0)

    return yaml_stream


def _walk_tree_and_resolve(key, current_node, tree, overrides, file_path):
    """A recursive function for resolving ``!ref`` and ``!copy`` tags.

    Loads additional yaml files if ``!include:`` tags are used.
    Also throws an error if ``!PLACEHOLDER`` tags are encountered.

    Arguments
    ---------
    key : str
        The fully-qualified path to current node.
    current_node : node
        A node in the yaml tree loaded with ruamel.yaml.
    tree : node
        The base node in the yaml tree loaded with ruamel.yaml.
    overrides : dict
        A set of overrides to pass to any ``!includes:`` files.
    file_path : str
        The location of the directory storing the main yaml file

    Returns
    -------
    yaml.Node
        A yaml tree with all references resolved.
    """

    # Walk sequence and resolve
    if isinstance(current_node, list):
        # print("list type")
        for i, sub_node in enumerate(current_node):
            sub_key = i if key == "root" else f"{key}[{i}]"
            # print("list subkey: {}".format(sub_key))
            current_node[i] = _walk_tree_and_resolve(
                sub_key, sub_node, tree, overrides, file_path
            )

    # Walk mapping and resolve.
    elif isinstance(current_node, dict):
        # print("dict type")
        for k, sub_node in current_node.items():
            # print("k: {}".format(k))
            sub_key = k if key == "root" else f"{key}[{k}]"
            # print("sub key: {}".format(sub_key))
            current_node[k] = _walk_tree_and_resolve(
                sub_key, sub_node, tree, overrides, file_path
            )

    # Base case, handle tags
    if hasattr(current_node, "tag"):
        tag_value = current_node.tag.value or ""
        # print("tag value: {}".format(tag_value))
        # Placeholders should have been replaced before now
        if tag_value == "!PLACEHOLDER":
            raise ValueError(f"'{key}' is a !PLACEHOLDER and must be replaced.")

        # Resolve references to other nodes
        elif tag_value in ["!ref", "!copy"]:
            # print("ref or copy")
            copy_mode = tag_value == "!copy"
            current_node = recursive_resolve(
                reference=current_node.value,
                reference_list=[],
                full_tree=tree,
                copy_mode=copy_mode,
            )

        # Include external yaml files
        elif tag_value.startswith("!include:"):
            # print("include")
            filename = tag_value[len("!include:") :]

            # Update overrides with child keys
            if isinstance(current_node, dict):
                if overrides:
                    recursive_update(overrides, current_node)
                else:
                    overrides = dict(current_node)

            if file_path is not None:
                filename = os.path.join(file_path, filename)
            with open(filename) as f:
                included_yaml = resolve_references(f, overrides)

            # Append resolved yaml to current node
            ruamel_yaml = ruamel.yaml.YAML()
            current_node = ruamel_yaml.load(included_yaml)

    # # Return node after all resolution is done.
    return current_node


def _make_tuple(loader, node):
    """Parse scalar node as a list, convert to tuple"""
    tuple_string = loader.construct_scalar(node)
    list_string = "[" + tuple_string[1:-1] + "]"
    parsed_list = yaml.load(list_string, Loader=yaml.Loader)
    return tuple(parsed_list)


def _load_node(loader, node):
    if isinstance(node, yaml.MappingNode):
        kwargs = loader.construct_mapping(node, deep=True)
        return [], kwargs
    elif isinstance(node, yaml.SequenceNode):
        args = loader.construct_sequence(node, deep=True)
        return args, {}
    return [], {}


def _construct_object(loader, callable_string, node):
    callable_ = pydoc.locate(callable_string)
    if callable_ is None:
        raise ImportError("There is no such class as %s" % callable_string)

    if not inspect.isclass(callable_):
        raise ValueError(
            f"!new:{callable_string} should be a class, but is {callable_}"
        )

    try:
        args, kwargs = _load_node(loader, node)
        return callable_(*args, **kwargs)
    except TypeError as e:
        err_msg = "Invalid argument to class %s" % callable_string
        e.args = (err_msg, *e.args)
        raise


def _construct_name(loader, callable_string, node):
    name = pydoc.locate(callable_string)
    if name is None:
        raise ImportError("There is no such entity as %s" % callable_string)

    if not (inspect.isclass(name) or inspect.isroutine(name)):
        args, kwargs = _load_node(loader, node)
        if args or kwargs:
            raise ValueError(
                f"!name:{callable_string} should be class or function, "
                f"if you specify args or kwargs. Instead it is {name}"
            )
        return name

    try:
        args, kwargs = _load_node(loader, node)
        return functools.partial(name, *args, **kwargs)
    except TypeError as e:
        err_msg = "Invalid argument to callable %s" % callable_string
        e.args = (err_msg, *e.args)
        raise


def _construct_module(loader, module_name, node):
    module = pydoc.locate(module_name)
    if module is None:
        raise ImportError("There is no such module as %s" % module_name)

    args, kwargs = _load_node(loader, node)
    if args != [] or kwargs != {}:
        raise ValueError("Cannot pass args to module")
    if not inspect.ismodule(module):
        raise ValueError(
            f"!module:{module_name} should be module, but is {module}"
        )

    return module


def _apply_function(loader, callable_string, node):
    callable_ = pydoc.locate(callable_string)
    if callable_ is None:
        raise ImportError("There is no such callable as %s" % callable_string)

    if not inspect.isroutine(callable_):
        raise ValueError(
            f"!apply:{callable_string} should be a callable, but is {callable_}"
        )

    try:
        args, kwargs = _load_node(loader, node)
        return callable_(*args, **kwargs)
    except TypeError as e:
        err_msg = "Invalid argument to callable %s" % callable_string
        e.args = (err_msg, *e.args)
        raise


def deref(ref, full_tree, copy_mode=False):
    """Find the value referred to by a reference in dot-notation

    Arguments
    ---------
    ref : str
        The location of the requested value, e.g. 'constants.param'
    full_tree : dict
        The dictionary to use for finding values
    copy_mode : bool
        Whether to copy the node before dereferencing.

    Returns
    -------
    node
        The node in the full_tree dictionary referenced by ``ref``.

    Example
    -------
    >>> deref('constants[a][b]', {'constants': {'a': {'b': 'c'}}})
    'c'
    """

    # Collect the attribute reference
    attr = None
    if "." in ref:
        ref, attr = ref.split(".", maxsplit=1)

    # Follow references in dot notation
    branch = full_tree
    for part in ref.split("["):
        part = part.strip("]")
        if part not in branch:
            raise ValueError('The reference "%s" is not valid' % ref)
        branch = branch[part]

    # Copy node if requested
    if copy_mode:
        return copy.deepcopy(branch)

    # To refer to an attribute, we add this special node
    if attr is not None:
        node = ruamel.yaml.comments.CommentedSeq()
        node += [branch, attr]
        node.yaml_set_tag("!apply:getattr")
        return node

    return branch


def recursive_resolve(reference, reference_list, full_tree, copy_mode=False):
    """Resolve a reference to a value, following chained references

    Arguments
    ---------
    reference : str
        a string containing '<x[y]>' in it where x[y] refers
        to a scalar node in the file.
    reference_list : list
        list of prior references in the chain, in order
        to catch circular references.
    full_tree : dict
        the dictionary in which to find all references and their values.
    copy_mode : bool
        Whether to perform a deep copy of the referenced node, rather than
        a shallow reference to the same object.

    Returns
    -------
    scalar
        The dereferenced value, with possible string interpolation and
        arithmetic parsing.

    Example
    -------
    >>> tree = {'a': 3, 'b': 'x', 'c': '<a>', 'd': '<c>/<c>', 'e': '<b>/<b>'}
    >>> recursive_resolve('<d>', [], tree)
    1.0
    >>> recursive_resolve('<e>', [], tree)
    'x/x'
    """
    # Non-greedy operator won't work here, because the fullmatch will
    # still match if the first and last things happen to be references
    reference_finder = re.compile(r"<[^>]*>")

    # Base case, no <key> present
    if not isinstance(reference, str) or not reference_finder.search(reference):
        return reference

    if len(reference_list) > 1 and reference in reference_list[1:]:
        raise ValueError("Circular reference detected: ", reference_list)

    # First check for a full match. These replacements preserve type.
    if reference_finder.fullmatch(reference):
        value = deref(reference.strip("<>"), full_tree, copy_mode)
        reference_list += [reference]
        return recursive_resolve(value, reference_list, full_tree, copy_mode)

    # Make sure reference list gets updated to prevent cycles
    matches = reference_finder.findall(reference)
    reference_list += [match[0] for match in matches]

    # Do replacements within the string (interpolation)
    def replace_fn(x, tree=full_tree, copy_mode=copy_mode):
        return str(deref(x[0].strip("<>"), full_tree=tree, copy_mode=copy_mode))

    sub = reference_finder.sub(replace_fn, reference)
    reference = recursive_resolve(sub, reference_list, full_tree, copy_mode)

    # Finally check for arithmetic operations.
    return parse_arithmetic(reference)


def parse_arithmetic(reference_string):
    """Parses simple arithmetic operations in references

    Adapted from https://stackoverflow.com/a/9558001/1761970

    Arguments
    ---------
    reference_string : str
        A string with references and possible arithmetic operations.

    Returns
    -------
    str
        Result of parsing and applying the arithmetic.

    Example
    -------
    >>> parse_arithmetic('2 * 6')
    12
    """
    try:
        return _ast_eval(ast.parse(reference_string, mode="eval").body)
    except (TypeError, SyntaxError, KeyError):
        return reference_string


def _ast_eval(node):
    ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Pow: op.pow,
        ast.Mod: op.mod,
    }
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return ops[type(node.op)](_ast_eval(node.left), _ast_eval(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return ops[type(node.op)](_ast_eval(node.operand))
    else:
        raise TypeError(node)


def recursive_update(d, u, must_match=False):
    """Similar function to `dict.update`, but for a nested `dict`.

    From: https://stackoverflow.com/a/3233356

    If you have to a nested mapping structure, for example:

        {"a": 1, "b": {"c": 2}}

    Say you want to update the above structure with:

        {"b": {"d": 3}}

    This function will produce:

        {"a": 1, "b": {"c": 2, "d": 3}}

    Instead of:

        {"a": 1, "b": {"d": 3}}

    Arguments
    ---------
    d : dict
        mapping to be updated
    u : dict
        mapping to update with
    must_match : bool
        Whether to throw an error if the key in `u` does not exist in `d`.

    Example
    -------
    >>> d = {'a': 1, 'b': {'c': 2}}
    >>> recursive_update(d, {'b': {'d': 3}})
    >>> d
    {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    # TODO: Consider cases where u has branch off k, but d does not.
    # e.g. d = {"a":1}, u = {"a": {"b": 2 }}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping) and k in d:
            recursive_update(d.get(k, {}), v)
        elif must_match and k not in d:
            raise KeyError(
                f"Override '{k}' not found in: {[key for key in d.keys()]}"
            )
        else:
            d[k] = v

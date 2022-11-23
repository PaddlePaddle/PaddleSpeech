"""A dependency graph for finding evaluation order.

Example
-------
>>> # The basic use case is that you have a bunch of keys
>>> # and some of them depend on each other:
>>> database = []
>>> functions = {'read': {'func': lambda: (0,1,2),
...                       'needs': []},
...              'process': {'func': lambda X: [x**2 for x in X],
...                          'needs': ['read']},
...              'save': {'func': lambda x: database.append(x),
...                       'needs': ['process']},
...              'print': {'func': lambda x,y: print(x, "became", y),
...                        'needs': ['read', 'process']},
...              'auxiliary': {'func': lambda: (1,2,3),
...                            'needs': []}}
>>> # If this is user supplied info, so you can't just hardcode the order,
>>> # a dependency graph may be needed.
>>> dg = DependencyGraph()
>>> # In simple cases, you can just encode the dependencies directly:
>>> for key, conf in functions.items():
...     for needed in conf["needs"]:
...         dg.add_edge(key, needed)
>>> # Now we can evaluate:
>>> outputs = {}
>>> for node in dg.get_evaluation_order():
...     f = functions[node.key]['func']
...     args = [outputs[needed] for needed in functions[node.key]['needs']]
...     outputs[node.key] = f(*args)
(0, 1, 2) became [0, 1, 4]
>>> # This added nodes implicitly.
>>> # However, since 'auxiliary' didn't depend on anything,
>>> # it didn't get added!
>>> assert 'auxiliary' not in outputs
>>> # So to be careful, we should also manually add nodes for any thing that
>>> # is not an intermediate step.
>>> _ = dg.add_node('auxiliary')
>>> assert 'auxiliary' in (node.key for node in dg.get_evaluation_order())
>>> # Arbitrary data can be added to nodes:
>>> dg2 = DependencyGraph()
>>> for key, conf in functions.items():
...     _ = dg2.add_node(key, conf)
...     for needed in conf["needs"]:
...         dg2.add_edge(key, needed)
>>> # Now we get access to the data in evaluation:
>>> outputs2 = {}
>>> for key, _, conf in dg2.get_evaluation_order():
...     f = conf['func']
...     args = [outputs[needed] for needed in conf['needs']]
...     outputs[key] = f(*args)
(0, 1, 2) became [0, 1, 4]

Authors:
    * Aku Rouhe 2020
"""
import collections
import uuid


class CircularDependencyError(ValueError):
    """
    An error caused by running into circular dependencies while searching for
    an evaluation order in a DependencyGraph.
    """

    pass


DGNode = collections.namedtuple("DGNode", ["key", "edges", "data"])
# A node in DependencyGraph.


class DependencyGraph:
    """General-purpose dependency graph.

    Essentially a directed acyclic graph.
    Usually used to find an evaluation order for e.g. variable substitution
    The relation that an edge between A and B represents is:
    "A depends on B, i.e. B should be evaluated before A"

    Nodes can be added explicitly or they can be created implicitly
    while adding edges.
    Nodes have keys, which should be some hashable value that identifies
    the elements the graph represents in your use case. E.G. they can just
    be the variable name you want to substitute.
    However, if needed, more generally you can attach any data to a node
    (e.g. a path in your tree), and if so desired, a unique key can be
    created for you. You'll only need to know that key while adding edges
    to/from it.
    Implicit keys and explicit keys can also be mixed.
    """

    def __init__(self):
        self.digraph = []
        self.key2ind = {}
        # Guard for manual duplicates (but not implicitly added ones)
        self._manually_added_keys = []

    @staticmethod
    def get_unique_key():
        """Returns a unique hashable identifier."""
        return uuid.uuid4()

    def add_node(self, key=None, data=None):
        """Adds a node explicitly.

        Arguments
        ---------
        key : hashable, optional
            If not given, a key is created for you.
        data : Any, optional
            Any additional data you wish to attach to this node.

        Returns
        -------
        hashable
            The key that was used (either yours or generated).

        Raises
        ------
        ValueError
            If node with the given key has already been added explicitly
            (with this method, not "add_edge").
        """
        if key is None:
            key = self.get_unique_key()
        elif key in self._manually_added_keys:
            raise ValueError("Adding duplicate node: {key}".format(key=key))
        else:
            self._manually_added_keys.append(key)
        if key in self.key2ind:  # Implicitly added already; don't add again.
            ind = self.key2ind[key]
            node = self.digraph[ind]
            # All that this operation can do is add data:
            self.digraph[ind] = DGNode(node.key, node.edges, data)
            return key
        self.key2ind[key] = len(self.digraph)
        self.digraph.append(DGNode(key, [], data))
        return key

    def add_edge(self, from_key, to_key):
        """Adds an edge, and implicitly also creates nodes for keys which have
        not been seen before. This will not let you add data to your nodes.
        The relation encodes: "from_key depends on to_key"
        (to_key must be evaluated before from_key).

        Arguments
        ---------
        from_key : hashable
            The key which depends on.
        to_key : hashable
            The key which is depended on.

        Returns
        -------
        None
        """
        from_ind = self._get_ind_and_add_if_new(from_key)
        to_ind = self._get_ind_and_add_if_new(to_key)
        edges_list = self.digraph[from_ind].edges
        if to_ind not in edges_list:
            edges_list.append(to_ind)

    def _get_ind_and_add_if_new(self, key):
        # Used internally to implicitly add nodes for unseen keys
        if key not in self.key2ind:
            self.key2ind[key] = len(self.digraph)
            self.digraph.append(DGNode(key, [], None))
        return self.key2ind[key]

    def is_valid(self):
        """Checks if an evaluation order can be found.

        A dependency graph is evaluatable if there are no circular
        dependencies, i.e., the graph is acyclic.

        Returns
        -------
        bool
            Indicating if the graph is evaluatable.
        """
        return not self._find_first_cycle()

    def get_evaluation_order(self, selected_keys=None):
        """Finds one valid evaluation order.

        There can be many different valid
        orders.
        NOTE: Generates output one DGNode at a time. May generate DGNodes
        before it finds a circular dependency. If you really need to know
        whether an order can be found, check is_valid() first. However,
        the algorithm for finding cycles is essentially the same as the one
        used for finding an evaluation order, so for very large graphs...
        Ah well, but maybe then you should be using some other solution
        anyway.

        Arguments
        ---------
        selected_keys : list, None
            List of keys. If not None, only the selected keys are guaranteed
            in the evaluation order (along with the keys they depend on).

        Yields
        ------
        DGNode
            The added DGNodes in a valid evaluation order.
            See the DGNode namedtuple above.

        Raises
        ------
        CircularDependencyError
            If a circular dependency is found.
        """
        seen_ever = set()

        def toposort(root_ind, visited):
            """Implementation of topsort."""
            nonlocal seen_ever
            here = visited + [root_ind]
            if root_ind in visited:
                raise CircularDependencyError(
                    "{cycle}".format(
                        cycle=" -> ".join(
                            str(self.digraph[i].key) for i in here
                        )
                    )
                )
            if root_ind in seen_ever:
                return  # Yield nothing
            seen_ever = seen_ever.union(set([root_ind]))
            for to_ind in self.digraph[root_ind].edges:
                for ind in toposort(to_ind, visited=here):
                    yield ind
            yield root_ind

        if selected_keys is None:
            start_inds = range(len(self.digraph))
        else:
            start_inds = [self.key2ind[key] for key in selected_keys]

        for start_ind in start_inds:
            for ind in toposort(start_ind, []):
                yield self.digraph[ind]

    def _find_first_cycle(self):
        """Depth-first search based algorithm for finding cycles in the graph."""
        seen_ever = set()

        def cycle_dfs(root_ind, visited):
            """Implementation of cycle_dfs."""
            nonlocal seen_ever
            print(root_ind, visited)
            here = visited + [root_ind]
            if root_ind in visited:
                return here
            if root_ind in seen_ever:
                return []
            seen_ever = seen_ever.union(set([root_ind]))
            for to_ind in self.digraph[root_ind].edges:
                cycle = cycle_dfs(to_ind, here)
                if cycle:
                    return cycle
            return []

        for ind in range(len(self.digraph)):
            if ind not in seen_ever:
                cycle = cycle_dfs(ind, [])
                if cycle:
                    return cycle
        return []

    def __contains__(self, key):
        # Allows the syntax:
        # 'key' in dependency_graph
        return key in self.key2ind

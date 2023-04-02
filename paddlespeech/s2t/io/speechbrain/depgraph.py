# Copyright (c) 2023 speechbrain Authors. All Rights Reserved.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from speechbrain 2023 (https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/utils/depgraph.py)
"""A dependency graph for finding evaluation order.

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
                raise CircularDependencyError("{cycle}".format(
                    cycle=" -> ".join(str(self.digraph[i].key) for i in here)))
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

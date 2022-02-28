import pytest


def test_dependency_graph():
    from speechbrain.utils.depgraph import (
        DependencyGraph,
        CircularDependencyError,
    )

    dg = DependencyGraph()
    # a->b->c
    dg.add_edge("b", "c")
    dg.add_edge("a", "b")
    assert dg.is_valid()
    eval_order = [node.key for node in dg.get_evaluation_order()]
    assert eval_order == ["c", "b", "a"]
    dg = DependencyGraph()
    # a->b->c, a->c
    dg.add_edge("b", "c")
    dg.add_edge("a", "b")
    dg.add_edge("a", "c")
    eval_order = [node.key for node in dg.get_evaluation_order()]
    assert eval_order == ["c", "b", "a"]
    dg = DependencyGraph()
    # a->b, a->c
    dg.add_edge("a", "b")
    dg.add_edge("a", "c")
    eval_order = [node.key for node in dg.get_evaluation_order()]
    assert eval_order == ["c", "b", "a"] or eval_order == ["b", "c", "a"]
    dg = DependencyGraph()
    # a->b, c->d
    dg.add_edge("a", "b")
    dg.add_edge("c", "d")
    eval_order = [node.key for node in dg.get_evaluation_order()]
    valid_orders = [
        ["d", "c", "b", "a"],
        ["d", "b", "c", "a"],
        ["d", "b", "a", "c"],
        ["b", "a", "d", "c"],
        ["b", "d", "a", "c"],
        ["b", "d", "c", "a"],
    ]
    assert eval_order in valid_orders
    dg = DependencyGraph()
    # a->b
    dg.add_node("a")
    dg.add_node("b")
    dg.add_edge("a", "b")
    eval_order = [node.key for node in dg.get_evaluation_order()]
    assert eval_order == ["b", "a"]
    dg = DependencyGraph()
    # a->b->a Impossible!
    dg.add_edge("a", "b")
    dg.add_edge("b", "a")
    assert not dg.is_valid()
    with pytest.raises(CircularDependencyError):
        list(dg.get_evaluation_order())
    dg = DependencyGraph()
    # a->b with data
    # should use uuids
    a_key = dg.add_node(data="a")
    assert a_key != "a"
    b_key = dg.add_node(data="b")
    dg.add_edge(a_key, b_key)
    eval_order_data = [node.data for node in dg.get_evaluation_order()]
    assert eval_order_data == ["b", "a"]
    # Adding same key in edge (implicitly) and then explicitly is ok:
    dg = DependencyGraph()
    dg.add_edge("a", "b")
    dg.add_node("a")
    eval_order = [node.key for node in dg.get_evaluation_order()]
    assert eval_order == ["b", "a"]
    # But adding same key twice explicitly will not work:
    with pytest.raises(ValueError):
        dg.add_node("a")

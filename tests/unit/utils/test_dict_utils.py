from fseval.utils.dict_utils import dict_flatten, dict_merge


def test_dict_flatten():
    nested = dict(user=dict(name="John"))
    flattened = dict_flatten(nested)
    assert flattened["user_name"] == "John"


def test_dict_merge():
    a = {"name": "John", "children": {"Elly": {"age": 3}}}
    b = {"age": 23, "children": {"Marie": {"age": 5}}}
    dict_merge(a, b)
    assert len(a["children"]) == 2
    assert "Elly" in a["children"].keys()
    assert "Marie" in a["children"].keys()
    assert a["age"] == 23

import itertools
from copy import deepcopy

from llm_food_taxonomy.graph.taxonomy import Taxonomy

dummy_tax = [
    ("food", "fruit"),
    ("food", "meat"),
    ("fruit", "apple"),
    ("fruit", "banana"),
    ("stone fruit", "pear"),
    ("fruit", "stone fruit"),
    ("stone fruit", "peach"),
    ("stone fruit", "apricot"),
    ("meat", "venison"),
    ("meat", "beef"),
    ("meat", "chicken"),
    ("meat", "pork"),
    ("food", "mushroom"),
    ("mushroom", "shiitake"),
    ("mushroom", "morel"),
    ("mushroom", "champignon"),
    ("food", "vegetable"),
    ("vegetable", "potato"),
    ("vegetable", "beetroot"),
    ("vegetable", "carrot"),
    ("vegetable", "yam"),
]

dummy_tax_small = dummy_tax[:12]


def get_node2name(tax):
    names = list(set(itertools.chain(*tax)))
    node2name = dict(zip(names, names))
    return node2name, names, deepcopy(names)


def get_dummy_taxonomy(rels=dummy_tax):
    node2name, names, ids = get_node2name(rels)
    node2name["pseudo root"] = "pseudo root"
    name2node = dict(zip(names, ids))
    return Taxonomy([(name2node[p], name2node[c]) for p, c in rels], node2name)

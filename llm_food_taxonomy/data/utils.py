import enum
from typing import Set, Tuple

import numpy as np
import pandas as pd


class RelationType(enum.Enum):
    PARENT_CHILD = "parent-child"
    ANCESTOR_DESCENDANT = "ancestor-descendant"


def breadth_first_search(node: dict, ancestors: tuple = ()):
    for k, v in node.items():
        if isinstance(v, dict) and len(v) > 0:
            yield from breadth_first_search(v, ancestors + (k,))
        else:
            yield ancestors + (k,)


def get_ancestry_df(tax, sep=",", id_sep=";", no_ids=False):
    ancestry = []
    ingredients = []
    for a in breadth_first_search(tax, ()):
        try:
            ancestry.append(sep.join(a) if sep is not None else a)
            ingredients.append(a[-1].strip())
        except TypeError as e:
            pass

    tax_df = pd.DataFrame({"leaf": ingredients, "ancestry": ancestry})

    def most_detailed(ancestries):
        idx = np.argmax([len(a.split(sep) if sep is not None else a) for a in ancestries])
        return ancestries.iloc[idx]

    tax_df = tax_df.groupby("leaf").agg({"ancestry": most_detailed}).reset_index(drop=False)
    tax_df["level"] = tax_df.ancestry.apply(lambda x: len(x.split(sep) if sep is not None else x))
    if no_ids:
        tax_df.ancestry = tax_df.ancestry.apply(lambda x: sep.join(e.split(id_sep)[-1] for e in x.split(sep)))
    return tax_df


def get_taxons(taxonomy: dict, accumulator: set = None) -> set:
    """
    Get all taxons in a taxonomy, meaning all node names in the taxonomy
    :param taxonomy: The taxonomy to traverse as a dictionary
    :param accumulator: An accumulator to store the taxons in
    :param make_ids: Whether to make ids for the taxons, useful for cases where the taxons are not unique
    :return: A set of all taxons in the taxonomy
    """
    if accumulator is None:
        accumulator = set()
    for k, v in taxonomy.items():
        accumulator.add(k)
        if isinstance(v, dict):
            get_taxons(v, accumulator)
    return accumulator


def get_branches(node: dict, accumulator: tuple = (), keep_ids=True):
    """
    Breadth first search to find all nodes in the dictionary
    :param accumulator: Accumulator for lineages in the tree
    :param node: Dict node to search
    :param keep_ids: Whether to keep the ids in the node names or not
    :return:
    """
    for k, v in node.items():
        if not keep_ids:
            k = tuple(k.split(":"))[-1]
        if isinstance(v, dict) and len(v) > 0:
            yield from get_branches(v, accumulator + (k,), keep_ids=keep_ids)
        else:
            yield accumulator + (k,)


def get_relations(taxonomy: dict,
                  accumulator: set = None,
                  relation=RelationType.PARENT_CHILD,
                  leaf_names: tuple = None,
                  keep_ids=True):
    """
    Get all parent-child relations in a taxonomy, meaning all node names in the taxonomy
    :param taxonomy: The taxonomy to traverse as a dictionary
    :param accumulator: An accumulator to store the taxons in
    :param relation: The type of relation to get, either parent-child or ancestor-descendant
    :param leaf_names: The leaf names to filter for getting the relations
    :param keep_ids: Whether to keep the ids in the node names or not
    :return: A set of parent-child relations in the taxonomy, where the first element is the parent,
             and the second element is the child
    """
    if not keep_ids:
        leaf_names = [tuple(l.split(":"))[-1] for l in leaf_names]
    branches = list(get_branches(taxonomy, (), keep_ids=keep_ids))
    if accumulator is None:
        accumulator: Set[Tuple[str, str]] = set()

    def _get_parent_child_pairs(branch: tuple):
        if leaf_names is None or branch[-1] in leaf_names:
            for i in range(len(branch) - 1):
                p, c = branch[i], branch[i + 1]
                if not keep_ids:
                    p, c = p.split(":")[-1], c.split(":")[-1]
                accumulator.add((p, c))

    def _get_ancestor_descendant_pairs(branch: tuple):
        if leaf_names is None or branch[-1] in leaf_names:
            for i in range(len(branch)):
                for j in range(i + 1, len(branch)):
                    a, d = branch[i], branch[j]
                    if not keep_ids:
                        a, d = a.split(":")[-1], d.split(":")[-1]
                    accumulator.add((a, d))

    for b in branches:
        if relation == RelationType.PARENT_CHILD:
            _get_parent_child_pairs(b)
        elif relation == RelationType.ANCESTOR_DESCENDANT:
            _get_ancestor_descendant_pairs(b)
        else:
            raise ValueError(f"Invalid relation type {relation}")

    return accumulator

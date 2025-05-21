import logging
from itertools import product
from pathlib import Path
from typing import Tuple, List, Set, Any, Dict

import networkx as nx
import numpy as np
from networkx import NetworkXError


class Taxonomy:
    """
    Wrapper for networkx with methods that are used repeatedly to build a Taxonomy
    I know this is ugly, don't hate me for it.
    """

    def __init__(self, relations: List[Tuple[str, str]], id_to_name: dict):
        self.g = nx.DiGraph(relations)
        self.pseudo_root = "pseudo root"
        nodes = set(self.g.nodes())
        for nid in id_to_name:
            if nid not in nodes:
                self.g.add_node(nid)
        self.id_to_name = id_to_name
        self.original_roots = self.roots()
        if self.pseudo_root not in nodes:
            self.g.add_node(self.pseudo_root)

    def insert(self, triplets: List[Tuple[str, str, str]]):
        """
        Insert new concepts into the taxonomy using triplets as placements (parent, query, child)
        :param triplets: A set of triplets with parent-child relations
        """
        new_relations = []
        removed_relations = []
        inserted_triplets = []
        it = triplets
        for p, q, c in it:
            t_removed_relations = []
            t_new_relations = []
            if (p, c) in self.g.edges():
                self.g.remove_edge(p, c)
                self.g.add_edge(p, q)
                self.g.add_edge(q, c)
                t_removed_relations.append((p, c))
                t_new_relations.append((p, q))
                t_new_relations.append((q, c))
            if c is not None and (q, c) not in self.g.edges():
                self.g.add_edge(q, c)
                t_new_relations.append((q, c))
            if p is not None and (p, q) not in self.g.edges():
                self.g.add_edge(p, q)
                t_new_relations.append((p, q))
            if len(t_new_relations) > 0:
                for n in [p, q, c]:
                    if n not in self.id_to_name and n is not None:
                        self.id_to_name[n] = n
        return new_relations, removed_relations, inserted_triplets

    def connect(self) -> None:
        roots = self.roots()
        if len(roots) > 1:
            if self.pseudo_root not in self.g.nodes():
                self.g.add_node(self.pseudo_root)
            incoming = list(self.g.predecessors(self.pseudo_root))
            assert len(incoming) == 0, f"Custom root has incoming edges: {incoming}"
            placements = [(None, self.pseudo_root, r) for r in roots if r != self.pseudo_root]
            self.insert(triplets=placements)
            incoming = list(self.g.predecessors(self.pseudo_root))
            assert len(incoming) == 0, f"Custom root has incoming edges: {incoming}"

    def roots(self):
        return [n for n in self.g.nodes() if self.g.in_degree(n) == 0]

    def depth(self):
        roots = [n for n in self.g.nodes() if self.g.in_degree(n) == 0]

        def _get_depth(node, accumulator=0):
            yield accumulator + 1
            edges = self.g.out_edges(node)
            for e in edges:
                yield from _get_depth(e[1], accumulator=accumulator + 1)

        depths = []
        try:
            for r in roots:
                depths.extend(list(_get_depth(r)))
        except RecursionError as re:
            raise re

        return max(depths)

    def node_children(self, node: Any):
        out_edges = self.g.out_edges(node)
        return tuple(map(lambda x: x[1], out_edges))

    def descendants(self, only_leaves=True) -> Dict[str, Set[str]]:
        acc = {}
        roots = [n for n in self.g.nodes() if self.g.in_degree(n) == 0]
        for r in roots:
            self.node_descendants(r, acc, only_leaves=only_leaves)
        return acc

    def node_descendants(self, node: Any, acc=None, only_leaves=True):
        if acc is None:
            acc = {}

        if node in acc:
            return acc[node]
        else:
            acc[node] = set()

        for n in self.node_children(node):
            if n in acc:
                continue
            if only_leaves and self.is_leaf(n) or not only_leaves:
                acc[node].add(n)
            if not self.is_leaf(n):
                acc[node] = acc[node].union(self.node_descendants(n, acc, only_leaves))
        return acc[node]

    def children(self) -> List[Tuple[str, Tuple[str, ...]]]:
        children = []
        for n in self.g.nodes():
            node_children = self.node_children(n)
            children.append((n, node_children))
        return children

    def inner_nodes(self):
        inner_nodes = [n for n in self.g.nodes() if self.g.out_degree(n) > 0]
        return inner_nodes

    def leaves(self):
        leaves = [n for n in self.g.nodes() if self.g.out_degree(n) == 0]
        return leaves

    def is_leaf(self, node):
        return self.g.out_degree(node) == 0

    def node_ancestry(self, node: str) -> List[str]:
        """
        Get all ancestries for a target node
        :param node: The node to get the ancestries for
        """
        g = self.g.reverse()

        def _get_ancestries(n, accumulator=()):
            edges = g.out_edges(n)
            if len(edges) == 0 or n in self.original_roots:
                return list(accumulator)
            for e in edges:
                if e[1] in self.original_roots and e[1] in accumulator:
                    continue
                if e[1] in accumulator:
                    logging.warning(f"Cycle detected for node {e[1]}")
                    return list(accumulator)
                return _get_ancestries(e[1], accumulator=accumulator + (e[1],))

        a = list(_get_ancestries(node, (node,)))
        assert a[0] == node, f"First node needs to be the target node ({node}): {a}"
        return a

    def ancestries(self, leaf_set=None) -> Set[List[str]]:
        """
        Get all ancestries for the taxonomy
        :param leaf_set: Allowed leaves for ancestries
        """

        roots = [n for n in self.g.nodes() if self.g.in_degree(n) == 0]

        def _get_ancestries(node, accumulator=()):
            yield list(accumulator + (node,))
            edges = self.g.out_edges(node)
            for e in edges:
                if e[1] in self.original_roots and e[1] in accumulator:
                    continue
                if e[1] in accumulator:
                    logging.warning(f"Cycle detected for node {e[1]}")
                    continue
                yield from _get_ancestries(e[1], accumulator=accumulator + (node,))

        a = []
        try:
            for r in roots:
                a.extend(list(_get_ancestries(r)))
        except RecursionError as re:
            raise re

        if leaf_set is not None:
            a = [l for l in a if l[-1] in leaf_set]

        return a

    def node_triplets(self, node: Any, existing=False):
        parents = list(self.g.predecessors(node))
        children = list(self.g.successors(node))
        if len(parents) == 0 or not existing:
            parents += [None]
        if len(children) == 0 or not existing:
            children += [None]
        for p, c in product(parents, children):
            if (p is None) and (c is None):
                continue
            yield p, node, c

    def triplets(self, existing=False) -> List[Tuple[str, str, str]]:
        """
        Get all triplets
        :param existing: Whether to only return triplets that already exist or all possible triplets
        """
        for n in self.g.nodes():
            yield from self.node_triplets(n, existing=existing)

    def queries_triplets(self, queries: List[str]):
        res = []
        for q in queries:
            res.append(list(self.query_triplets(q)))
        return res

    def query_triplets(self, query_node):
        try:
            return list(self.node_triplets(query_node, existing=True))
        except NetworkXError as ne:
            print("NetworkX error: {}".format(ne))
            return []

    def most_specific(self, nodes):
        try:
            ancestries = [tuple(reversed(self.node_ancestry(n))) for n in nodes]
            nodes = [a[-1] for a in ancestries
                     if not np.any([a == oa[:len(a)] for oa in ancestries if a != oa])]
        except NetworkXError as ne:
            logging.warning(f"Error occured when trying to get most specific nodes: {str(ne)}")
        return nodes

    def most_general(self, nodes):
        try:
            ancestries = [tuple(reversed(self.node_ancestry(n))) for n in nodes]
            nodes = [a[-1] for a in ancestries
                     if not np.any([a[:len(oa)] == oa for oa in ancestries if a != oa])]
        except NetworkXError as ne:
            logging.warning(f"Error occured when trying to get most general nodes: {str(ne)}")
        return nodes

    def save(self, directory_path: str):
        path = Path(directory_path)
        name = path.name
        with open(path / f'{name}.terms', 'w') as f:
            for nid, nname in self.id_to_name.items():
                f.write(f"{nid}\t{nname}\n")
        with open(path / f'{name}.desc', 'w') as f:
            for nid, nname in self.id_to_name.items():
                f.write(f"{nid}\t{nname}\n")
        with open(path / f'{name}.taxo', 'w') as f:
            for p, c in self.g.edges():
                f.write(f"{p}\t{c}\n")

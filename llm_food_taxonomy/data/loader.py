import glob
import itertools
import json
from collections import deque
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx import descendants

from llm_food_taxonomy.util import load_json

pseudo_root = "pseudo root"
pseudo_leaf = "pseudo leaf"


def _clean_semeval_verb(node_name):
    if node_name is not None and node_name != "":
        return node_name.split("||")[0].split(".")[0].replace("_", " ")
    return node_name


def _clean_mesh(node_name):
    if node_name is not None and node_name != "":
        return " ".join(node_name.replace(",", "").split())
    return node_name


def _find_insert_position(node_ids, core_subgraph, holdout_graph):
    node2pos = {}
    subgraph = core_subgraph
    for node in node_ids:
        parents = set()
        children = set()
        ps = deque(holdout_graph.predecessors(node))
        cs = deque(holdout_graph.successors(node))
        while ps:
            p = ps.popleft()
            if p in subgraph:
                parents.add(p)
            else:
                ps += list(holdout_graph.predecessors(p))
        while cs:
            c = cs.popleft()
            if c in subgraph:
                children.add(c)
            else:
                cs += list(holdout_graph.successors(c))
        if not children:
            children.add(pseudo_leaf)
        position = [(p, c) for p in parents for c in children if p != c]
        node2pos[node] = position
    return node2pos


def _remove_jump_edges(subgraph: nx.DiGraph):
    # remove jump edges
    node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
    for node in subgraph.nodes():
        if subgraph.out_degree(node) > 1:
            successors1 = set(subgraph.successors(node))
            successors2 = set(itertools.chain.from_iterable([node2descendants[n] for n in successors1]))
            checkset = successors1.intersection(successors2)
            if checkset:
                for s in checkset:
                    if subgraph.in_degree(s) > 1:
                        subgraph.remove_edge(node, s)


def _get_holdout_subgraph(nodes, full_graph):
    node_to_remove = [n for n in full_graph.nodes if n not in nodes]
    subgraph = full_graph.subgraph([node for node in nodes]).copy()
    for node in node_to_remove:
        parents = set()
        children = set()
        ps = deque(full_graph.predecessors(node))
        cs = deque(full_graph.successors(node))
        while ps:
            p = ps.popleft()
            if p in subgraph:
                parents.add(p)
            else:
                ps += list(full_graph.predecessors(p))
        while cs:
            c = cs.popleft()
            if c in subgraph:
                children.add(c)
            else:
                cs += list(full_graph.successors(c))
        for p, c in itertools.product(parents, children):
            subgraph.add_edge(p, c)
    _remove_jump_edges(subgraph)
    return subgraph


def _get_split(terms: pd.DataFrame,
               taxonomy_dir: str,
               split_name="train"):
    split_file = glob.glob(str(Path(taxonomy_dir) / f"*terms.{split_name}"))[0]

    with open(split_file) as f:
        split_line_numbers = f.readlines()

    split_terms = terms.iloc[np.array(split_line_numbers, dtype=int)].node_id.values
    return split_terms


def _remove_mesh_cycles(taxo: pd.DataFrame):
    taxo = taxo[taxo.apply(lambda r: not ((r.hyponym == "proteins")
                                          and (r.hypernym in ['glycoproteins',
                                                              'bloodproteins'])), axis=1)]
    return taxo


def add_pseudo_nodes(g: nx.DiGraph, add_to_all: bool = False):
    g.add_node(pseudo_leaf)
    for node in g.nodes():
        if g.out_degree(node) == 0 or add_to_all:
            g.add_edge(node, pseudo_leaf)
    return g


def get_subgraphs(taxos, all_seed_nodes, all_non_seed_nodes):
    g = nx.DiGraph(taxos[["hypernym", "hyponym"]].values.tolist())
    all_nodes = np.union1d(all_seed_nodes, all_non_seed_nodes)
    roots = [node for node in g.nodes() if g.in_degree(node) == 0]
    g.add_node(pseudo_root)
    for node in roots:
        g.add_edge(pseudo_root, node)
    all_seed_nodes.append(pseudo_root)

    for node in all_nodes:
        if node not in g.nodes():
            g.add_node(node)
    core_subgraph = _get_holdout_subgraph(all_seed_nodes, g)
    core_subgraph = add_pseudo_nodes(core_subgraph, add_to_all=True)
    holdout_subgraph = _get_holdout_subgraph(all_nodes, g)
    holdout_subgraph = add_pseudo_nodes(holdout_subgraph, add_to_all=False)
    return core_subgraph, holdout_subgraph


def get_positions(terms: pd.DataFrame, taxos: pd.DataFrame, split="val"):
    """
    If splitting out a node, connect its parent and child nodes together, such that we have a valid tree
    :return:
    """
    all_seed_nodes = terms[terms.split == "train"].node_id.unique().tolist()
    all_non_seed_nodes = terms[terms.split == split].node_id.unique().tolist()

    core_subgraph, holdout_subgraph = get_subgraphs(taxos,
                                                    all_seed_nodes=all_seed_nodes,
                                                    all_non_seed_nodes=all_non_seed_nodes)
    positions = _find_insert_position(all_non_seed_nodes,
                                      core_subgraph=core_subgraph,
                                      holdout_graph=holdout_subgraph)
    return positions, core_subgraph


def get_split_name(row, val_node_ids, test_node_ids):
    if row.node_id in val_node_ids:
        return "val"
    elif row.node_id in test_node_ids:
        return "test"
    else:
        return "train"


def _load_base_taxonomy(taxonomy_dir: str, with_embeddings: bool = False, with_split=True):
    terms_file = glob.glob(str(Path(taxonomy_dir) / "*.terms"))[0]
    taxo_file = glob.glob(str(Path(taxonomy_dir) / "*.taxo"))[0]
    desc_file = glob.glob(str(Path(taxonomy_dir) / "*.desc"))[0]

    with open(terms_file, "r") as f:
        term_lines = f.readlines()

    with open(taxo_file, "r") as f:
        taxo_lines = f.readlines()

    with open(desc_file, "r") as f:
        desc_lines = f.readlines()

    terms = np.array([t.strip().split("\t") for t in term_lines], dtype=str).T
    taxos = np.array([t.strip().split("\t") for t in taxo_lines], dtype=str)
    descs = np.array([t.strip().split("\t") for t in desc_lines], dtype=str).T

    terms = pd.DataFrame({"node_id": terms[0],
                          "node_name": terms[1]})

    descs = pd.DataFrame({"node_name": descs[0],
                          "desc": descs[1]})
    terms = pd.merge(terms, descs, left_on="node_name", right_on="node_name", how="left")
    terms.desc = terms.desc.fillna(terms.node_name)
    terms.node_name = terms.node_name.apply(str.lower)

    terms["unique_node_name"] = terms["node_name"]
    taxos = pd.DataFrame(taxos, columns=["hypernym", "hyponym"])

    if "semeval_verb" in taxonomy_dir:
        terms["node_name"] = terms.node_name.apply(_clean_semeval_verb)

    if "mesh" in taxonomy_dir:
        terms["node_name"] = terms.node_name.apply(_clean_mesh)
        taxos = _remove_mesh_cycles(taxos)

    if with_embeddings:
        embed_file = glob.glob(str(Path(taxonomy_dir) / "*terms.embed"))[0]
        with open(embed_file, "r") as f:
            emb_lines = f.readlines()
        embs = np.stack([t.strip().split(" ") for t in emb_lines[1:]], dtype=str)
        print(len(embs[:, 0]))
        print(len(list(embs[:, 1:])))
        embs = pd.DataFrame({"node_id": embs[:, 0],
                             "embedding": embs[:, 1:].tolist()})

        terms = pd.merge(terms, embs, left_on="node_id", right_on="node_id")

    if with_split:
        _get_split_partial = partial(_get_split,
                                     terms,
                                     taxonomy_dir)
        val_node_ids = _get_split_partial(split_name="validation")
        test_node_ids = _get_split_partial(split_name="test")

        terms["split"] = terms.apply(partial(get_split_name,
                                             val_node_ids=val_node_ids,
                                             test_node_ids=test_node_ids), axis=1)

    terms.drop_duplicates(subset=["node_id"], keep="first", inplace=True)
    return terms, taxos


def load_subgraph_taxonomy(taxonomy_dir: str, mode: str = 'test'):
    terms, taxos = _load_base_taxonomy(taxonomy_dir, with_embeddings=True, with_split=True)
    terms = terms[(terms.split == "train") | (terms.split == mode)]

    node2pos = {}
    name_lookup = terms.set_index("node_id")
    for split in ["train", mode]:
        positions, core_subgraph = get_positions(terms, taxos, split=split)
        node2pos.update(positions)

    terms["positions"] = terms.node_id.apply(lambda nid: node2pos[nid])
    terms.positions = terms.positions.progress_apply(lambda p: list(set(
        [tuple([name_lookup.loc[nid].node_name
                if (nid != pseudo_root) and (nid != pseudo_leaf) else None
                for nid in pos])
         for pos in p])))
    terms["leaf"] = terms["positions"].apply(lambda pos: all(p[1] is None for p in pos))
    taxos = pd.DataFrame(list(core_subgraph.edges()), columns=["hypernym", "hyponym"])
    return terms, taxos


def load_taxonomy(taxonomy_dir: str, with_split=False, with_embeddings=False):
    """
    Load a taxonomy from a directory containing a .terms and .taxo file
    """
    terms, taxos = _load_base_taxonomy(taxonomy_dir, with_embeddings=with_embeddings, with_split=with_split)

    if with_split:
        node2pos = {}
        name_lookup = terms.set_index("node_id")
        for split in ["train", "val", "test"]:
            positions, _ = get_positions(terms, taxos, split)
            node2pos.update(positions)

        terms["positions"] = terms.node_id.apply(lambda nid: node2pos.get(nid, []))
        terms.positions = terms.positions.progress_apply(lambda p: list(set(
            [tuple([name_lookup.loc[nid].node_name
                    if (nid != pseudo_root) and (nid != pseudo_leaf) else None
                    for nid in pos])
             for pos in p])))
        terms["leaf"] = terms["positions"].apply(lambda pos: all(p[1] is None for p in pos))
    return terms, taxos


def flatten_dict(dd, separator='_', prefix=''):
    """Flatten a nested dictionary with keys concatenated by a separator."""
    res = {}
    for key, value in dd.items():
        # Create new key by concatenating prefix and current key
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            # Recursively flatten the dictionary
            res.update(flatten_dict(value, separator, new_key))
        else:
            res[new_key] = value
    return res


def load_completion(taxonomy_dir: str, streamed=False, with_reasoning=False):
    """
    Load the completion results from the output directory
    :param taxonomy_dir: The directory containing the output files
    :param streamed: Whether the completion was streamed
    :param with_reasoning: Whether to load the reasoning data (model generated text)
    :return: The terms and taxo dataframes
    """
    terms_file = glob.glob(str(Path(taxonomy_dir) / "*.terms"))[0]
    try:
        taxo_file = glob.glob(str(Path(taxonomy_dir) / "*.triplets"))[0]
        triplets = pd.read_csv(taxo_file, sep="\t")
        triplets.columns = ["query_node", "parent", "child"]
        triplets = triplets.replace({np.nan: None})
        triplets["predicted_positions"] = triplets.apply(lambda row: (row.parent, row.child), axis=1)
        triplets = triplets.drop(columns=["parent", "child"])
        triplets = triplets.groupby("query_node").agg(list).reset_index(drop=False)
    except:
        pos_file = glob.glob(str(Path(taxonomy_dir) / "*.pos.json"))[0]
        if not streamed:
            positions = load_json(pos_file)
        else:
            positions = {}
            with open(pos_file, "r") as f:
                for l in f.readlines():
                    pos = json.loads(l)
                    if pos is not None:
                        positions.update(pos)
        triplets = list(itertools.chain(*[[(p, q, c) for p, c in v] for q, v in positions.items()]))
        triplets = pd.DataFrame({
            "query_node": [t[1] for t in triplets],
            "predicted_positions": [(t[0], t[2]) for t in triplets]
        })
        triplets = triplets.groupby("query_node").agg(list).reset_index(drop=False)
        triplets = triplets[triplets.predicted_positions.apply(lambda x: x is not None and len(x) > 0)]
    with open(terms_file, "r") as f:
        term_lines = f.readlines()

    terms = np.array([t.strip().split("\t") for t in term_lines], dtype=str).T

    terms = pd.DataFrame({"node_id": terms[0],
                          "node_name": terms[1]})

    if "semeval_verb" in str(taxonomy_dir):
        terms["node_name"] = terms.node_name.apply(_clean_semeval_verb)
        triplets.query_node = triplets.query_node.apply(_clean_semeval_verb)
        triplets.predicted_positions = triplets.predicted_positions.apply(
            lambda x: [tuple([_clean_semeval_verb(n) for n in t]) for t in x]
        )

    if "mesh" in str(taxonomy_dir):
        terms["node_name"] = terms.node_name.apply(_clean_mesh)
        triplets.query_node = triplets.query_node.apply(_clean_mesh)
        triplets.predicted_positions = triplets.predicted_positions.apply(
            lambda x: [tuple([_clean_mesh(n) for n in t]) for t in x]
        )

    if with_reasoning:
        reason_file = glob.glob(str(Path(taxonomy_dir) / "predictions.outputs.json"))
        if len(reason_file) == 0:
            return terms, triplets

        with open(reason_file[0], "r") as f:
            reasons = json.load(f)
        flat_reasons = [pd.Series([r, *flatten_dict(v).values()],
                                  index=["query_node", *flatten_dict(v).keys()])
                        for r, v in reasons.items()]
        reasons_df = pd.DataFrame(flat_reasons)
        triplets = pd.merge(triplets, reasons_df, left_on="query_node", right_on="query_node", how="left")
    return terms, triplets


def load_baseline_taxonomy(taxonomy_dir: str):
    df = pd.read_csv(taxonomy_dir, sep="\t")
    df.columns = df.columns.str.lower()
    df.child = df.child.apply(lambda x: None if x == "" else x)
    df.parent = df.parent.apply(lambda x: None if x == "" else x)
    df = df[['query', 'parent', 'child']]
    if "semeval_verb" in str(taxonomy_dir):
        for c in df.columns:
            df[c] = df[c].apply(_clean_semeval_verb)

    if "mesh" in str(taxonomy_dir):
        for c in df.columns:
            df[c] = df[c].apply(_clean_mesh)
    return df

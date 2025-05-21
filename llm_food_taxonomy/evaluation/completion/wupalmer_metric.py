from typing import Set, List, Dict, Tuple, Any, Optional

import numpy as np

from llm_food_taxonomy.evaluation.metric import ScoreAccumulator, Metric
from llm_food_taxonomy.graph.taxonomy import Taxonomy


class WuPSimilarity(Metric):

    @staticmethod
    def similarity(ancestry1: list[str], ancestry2: list[str]) -> float:
        least_common_ancestor = 0
        path_intersection = set(ancestry1).intersection(set(ancestry2))

        for ancestor in path_intersection:
            depth = min(ancestry1.index(ancestor), ancestry2.index(ancestor))
            least_common_ancestor = max(least_common_ancestor, depth)
        least_common_ancestor += 1

        divisor = len(ancestry1) + len(ancestry2)
        if divisor == 0:
            score = np.nan
        else:
            score = (2 * least_common_ancestor) / divisor
        return score

    @staticmethod
    def get_ancestries(tax: Taxonomy,
                       node2name: dict[Any, str],
                       leaf_set: Optional[Set[str]] = None,
                       ) -> Dict[str, Tuple[str, ...]]:
        return {a[-1]: tuple(a)
                for a in tax.ancestries(leaf_set=leaf_set)}

    @classmethod
    def calculate(cls,
                  pred_positions: Dict[str, List[Tuple[str, str]]],
                  true_positions: Dict[str, List[Tuple[str, str]]],
                  node2name: Dict[str, str],
                  seed_taxonomy: List[Tuple[str, str]],
                  leaves: Set[str] = None,
                  verbose=False) -> (ScoreAccumulator, ScoreAccumulator):
        """
        Calculate the wu-palmer similarity metric
        :param pred_positions: Predicted positions (parent, child) added to the taxonomy
        :param true_positions: Gold standard relations (parent, child) in the test data
        :param seed_taxonomy: The seed taxonomy
        :param node2name: Mapping from node id to node name for the true taxonomy
        :param leaves: The set of leaf node ids
        :param verbose: Print the relations
        :return:
        """
        if leaves is None:
            leaves = set()
        queries = set(true_positions.keys())
        name2node = dict(zip(node2name.values(), node2name.keys()))
        query_ids = set([name2node[n] for n in queries])

        pred_triplets = []
        true_triplets = []

        for q, positions in true_positions.items():
            for p, c in positions:
                true_triplets.append((p, q, c))

        for q, positions in pred_positions.items():
            for p, c in positions:
                pred_triplets.append((p, q, c))

        pred_tax = Taxonomy(seed_taxonomy, id_to_name=node2name)
        pred_tax.insert(pred_triplets)

        true_tax = Taxonomy(seed_taxonomy, id_to_name=node2name)
        true_tax.insert(true_triplets)

        # If we only use the new relations, we will have a disconnected graph, thus we need to add the seed relations
        # And then we need to remove the seed ancestor relations from the new relations
        pred_ancestries = cls.get_ancestries(pred_tax, leaf_set=query_ids, node2name=node2name)
        true_ancestries = cls.get_ancestries(true_tax, leaf_set=query_ids, node2name=node2name)

        scores = []
        leaf_scores = []
        nonleaf_scores = []
        for q in queries:
            true = true_ancestries.get(q, tuple())
            pred = pred_ancestries.get(q, tuple())

            score = cls.similarity(true, pred)
            scores.append(score)
            if q in leaves:
                leaf_scores.append(score)
            else:
                nonleaf_scores.append(score)

        return ({"WPS": np.nanmean(scores), "scores": np.array(scores)},
                {"WPS": np.nanmean(nonleaf_scores), "scores": np.array(nonleaf_scores)},
                {"WPS": np.nanmean(leaf_scores), "scores": np.array(leaf_scores)})

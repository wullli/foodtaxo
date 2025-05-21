from typing import List, Tuple, Dict
import numpy as np

from llm_food_taxonomy.evaluation.metric import UnsupervisedMetric
from llm_food_taxonomy.graph.taxonomy import Taxonomy


class UnclassifiedNodes(UnsupervisedMetric):

    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  subset=None,
                  similarity_map: dict = None) -> float:
        tax = Taxonomy(taxonomy_relations, node2name)
        return len([n for n in tax.g.nodes
                    if (tax.g.out_degree(n) == 0)
                    and (tax.g.in_degree(n) == 0)]) / len(tax.g.nodes)


class HumanMemoryHeuristic(UnsupervisedMetric):

    def __init__(self, threshold: float):
        self.threshold = threshold

    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  subset=None,
                  similarity_map: dict = None) -> float:

        tax = Taxonomy(taxonomy_relations, node2name)
        error_counter = 0
        nodes = tax.children()

        for _, children in nodes:
            if len(children) > self.threshold:
                error_counter += 1

        return 1 - (error_counter / max(1, len(nodes)))


class AmountAndDepthOfConstructs(UnsupervisedMetric):

    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  subset=None,
                  similarity_map: dict = None) -> float:
        tax = Taxonomy(taxonomy_relations, node2name)
        tax.connect()
        all_ancestries = tax.ancestries()
        depth_score = 0
        for ancestors in all_ancestries[
                         1:]:  # Calculate the depth score for only categories and characteristics, excluding the root.
            if len(ancestors) > 1:
                depth_score += 1 / (len(ancestors) - 1)  # Minus one, because each node is its own ancestor.

        return 1 / (np.log(depth_score - 1) + 1)


class BranchingFactor(UnsupervisedMetric):
    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  subset=None,
                  similarity_map: dict = None) -> float:
        tax = Taxonomy(taxonomy_relations, node2name)
        tax.connect()
        nodes = tax.children()
        return np.mean([len(cs) for _, cs in nodes])

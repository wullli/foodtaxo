from typing import Set, List, Dict, Tuple

from llm_food_taxonomy.evaluation.metric import ScoreAccumulator, Metric


class ParentMetric(Metric):

    @classmethod
    def calculate(cls,
                  pred_positions: Dict[str, List[Tuple[str, str]]],
                  true_positions: Dict[str, List[Tuple[str, str]]],
                  node2name: Dict[str, str],
                  seed_taxonomy: List[Tuple[str, str]],
                  leaves: Set[str] = None,
                  verbose=False,
                  first_only=True) -> (Dict[str, float], Dict[str, float]):
        """
        Calculate the ancestor-f1 metric as described in the paper
        :param pred_positions: Predicted relations (parent, child) added to the taxonomy
        :param true_positions: Gold standard relations (parent, child) in the test data
        :param node2name: Mapping from node id to node name for the true taxonomy
        :param seed_taxonomy: Mapping from node id to node name for the true taxonomy
        :param leaves: The set of leaf node ids
        :param verbose: Print verbose output
        :return:
        """
        if leaves is None:
            leaves = set()
        queries = set(true_positions.keys())
        acc = ScoreAccumulator()
        leaf_acc = ScoreAccumulator()
        nonleaf_acc = ScoreAccumulator()
        for query in queries:
            truth = [p[0] for p in true_positions.get(query, [])]
            if len(truth) == 0:
                print(f"Query {query} has no truth!")
                continue

            pred = [p[0] for p in pred_positions.get(query, [])]

            if first_only:
                pred = pred[:1]

            tp = len(set(truth).intersection(set(pred)))
            fp = len(set(pred).difference(set(truth)))
            fn = len(set(truth).difference(set(pred)))

            acc.tp += tp
            acc.fp += fp
            acc.fn += fn

            if query in leaves:
                leaf_acc.tp += tp
                leaf_acc.fp += fp
                leaf_acc.fn += fn
            else:
                nonleaf_acc.tp += tp
                nonleaf_acc.fp += fp
                nonleaf_acc.fn += fn

        return ({"F1": acc.f1(), "Precision": acc.precision(), "Recall": acc.recall()},
                {"F1": nonleaf_acc.f1(), "Precision": nonleaf_acc.precision(), "Recall": nonleaf_acc.recall()},
                {"F1": leaf_acc.f1(), "Precision": leaf_acc.precision(), "Recall": leaf_acc.recall()})

import abc
import unittest
from typing import Optional

from llm_food_taxonomy.evaluation import Metric, PositionMetric
from llm_food_taxonomy.evaluation.completion.parent_metric import ParentMetric


class _TestMetric(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.metric: Optional[Metric] = None
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def setUp(self):
        raise NotImplementedError("Subclasses must implement setUp method")

    def test_score_one(self):
        pred_relations = [("b", "d"), ("c", "e")]
        true_relations = [("b", "d"), ("c", "e")]
        seed_relations = [("a", "b"), ("a", "c")]
        id_to_name = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}
        acc, lacc = self.metric.calculate(pred_relations,
                                          true_relations,
                                          seed_relations,
                                          concepts_to_add={"d", "e"},
                                          node2name=id_to_name,
                                          leaves={"d", "e"})
        print(acc.f1())
        self.assertEqual(acc.f1(), 1.0)

    def test_leaf_score_one(self):
        pred_relations = [("b", "d"), ("c", "e")]
        true_relations = [("b", "d"), ("c", "e")]
        seed_relations = [("a", "b"), ("a", "c")]
        id_to_name = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}
        acc, lacc = self.metric.calculate(pred_relations,
                                          true_relations,
                                          seed_relations,
                                          concepts_to_add={"d", "e"},
                                          node2name=id_to_name,
                                          leaves={"d", "e"})
        print(lacc.f1())
        self.assertEqual(lacc.f1(), 1.0)

    def test_score_zero(self):
        pred_relations = [("b", "d"), ("c", "e")]
        true_relations = [("d", "b"), ("e", "c")]
        seed_relations = [("a", "b"), ("a", "c")]
        id_to_name = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}
        acc, lacc = self.metric.calculate(pred_relations,
                                          true_relations,
                                          seed_relations,
                                          concepts_to_add={"d", "e"},
                                          node2name=id_to_name,
                                          leaves={"d", "e"})
        print(acc.f1())
        self.assertEqual(acc.f1(), 0.0)

    def test_leaf_score_zero(self):
        pred_relations = [("b", "d"), ("c", "e")]
        true_relations = [("d", "b"), ("e", "c")]
        seed_relations = [("a", "b"), ("a", "c")]
        id_to_name = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}
        acc, lacc = self.metric.calculate(pred_relations,
                                          true_relations,
                                          seed_relations,
                                          concepts_to_add={"d", "e"},
                                          node2name=id_to_name,
                                          leaves={"d", "e"})
        print(lacc.f1())
        self.assertEqual(lacc.f1(), 0.0)

    def test_score_middle(self):
        pred_relations = [("b", "d"), ("c", "e")]
        true_relations = [("b", "d"), ("e", "c")]
        seed_relations = [("a", "b"), ("a", "c")]
        id_to_name = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}
        acc, lacc = self.metric.calculate(pred_relations,
                                          true_relations,
                                          seed_relations,
                                          concepts_to_add={"d", "e"},
                                          node2name=id_to_name,
                                          leaves={"d", "e"})
        print(acc.f1())
        self.assertGreater(acc.f1(), 0.0)
        self.assertGreater(1.0, acc.f1())

    def test_leaf_score_middle(self):
        pred_relations = [("b", "d"), ("c", "e")]
        true_relations = [("b", "d"), ("e", "c")]
        seed_relations = [("a", "b"), ("a", "c")]
        id_to_name = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}
        acc, lacc = self.metric.calculate(pred_relations,
                                          true_relations,
                                          seed_relations,
                                          node2name=id_to_name,
                                          concepts_to_add={"d", "e"},
                                          leaves={"d", "e"})
        print(lacc.f1())
        self.assertGreater(lacc.f1(), 0.0)
        self.assertGreater(1.0, lacc.f1())


class TestAncestorMetric(_TestMetric):
    def setUp(self):
        self.metric = ParentMetric()


class TestParentMetric(_TestMetric):
    def setUp(self):
        self.metric = PositionMetric()

    def test_score_middle(self):
        pred_relations = [('Fruit vegetables', 'McCormick Italian Herb Grinder'),
                          ('Confectionery', 'Chocolate Biscuit'),
                          ('Sugar Alternatives', 'Monosodium glutamate'),
                          ('Veal', 'Veal Cutlets From The Nut'),
                          ('Root vegetables', 'Karma Vegetable Empanada'),
                          ('Food', 'Milk')]
        true_relations = [('Herbs & Spices', 'McCormick Italian Herb Grinder'),
                          ('Confectionery', 'Chocolate Biscuit'),
                          ('Sugar Alternatives', 'Monosodium glutamate'),
                          ('Veal', 'Veal Cutlets From The Nut'),
                          ('Meals', 'Karma Vegetable Empanada'),
                          ('Dairy', 'Milk')]
        seed_relations = []
        for p, c in true_relations:
            seed_relations.append(('Food', p))


        id_to_name = {i: i for i in all_set}
        concepts_to_add = {c for _, c in true_relations}
        acc, lacc = self.metric.calculate(pred_relations,
                                          true_relations,
                                          seed_taxonomy=seed_relations,
                                          node2name=id_to_name,
                                          leaves=concepts_to_add)
        print(lacc.f1())
        self.assertGreater(lacc.f1(), 0.0)
        self.assertGreater(1.0, lacc.f1())


del _TestMetric

import abc
import itertools
import unittest
from copy import deepcopy
from random import Random
from typing import Optional

import pytest

from llm_food_taxonomy.evaluation.generation.conciseness import *
from llm_food_taxonomy.evaluation.generation.robustness import *
from test.stubs import dummy_tax, get_node2name, get_dummy_taxonomy


class _TestMetric(unittest.TestCase):
    fasttext = fasttext.load_model("../cc.en.300.bin")

    def __init__(self, *args, **kwargs):
        self.metric: Optional[UnsupervisedMetric] = None
        self.maximize = True

        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def setUp(self):
        raise NotImplementedError("Subclasses must implement setUp method")

    @pytest.mark.robustness
    def test_robustness_shuffle(self):

        node2name, names, ids = get_node2name(dummy_tax)
        name2node = dict(zip(names, ids))
        Random(123).shuffle(ids)
        shuffled_node2name = dict(zip(ids, names))

        tax1 = [(name2node[p], name2node[c]) for p, c in dummy_tax]
        tax2 = [(name2node[p], name2node[c]) for p, c in dummy_tax]

        normal_tax_score = self.metric.calculate(tax1, node2name)
        shuffled_tax_score = self.metric.calculate(tax2, shuffled_node2name)

        print("Shuffled tax score {}".format(shuffled_tax_score))
        print("Normal tax score {}".format(normal_tax_score))

        if self.maximize:
            self.assertTrue(normal_tax_score > shuffled_tax_score)
        else:
            self.assertTrue(normal_tax_score < shuffled_tax_score)

    @staticmethod
    def shuffle_leaf_sets(rnd: Random, taxonomy: Taxonomy, node2name: dict):
        new_relations = deepcopy(taxonomy.relations)
        all_leaves = taxonomy.leaves()
        children_sets = [(p, set(c).intersection(all_leaves)) for p, c in taxonomy.children()
                         if len(set(c).intersection(all_leaves)) > 1]
        parents = [p for p, c in children_sets]

        chosen_parents = deepcopy(parents)
        rnd.shuffle(chosen_parents)

        for i, (p, children) in enumerate(children_sets):
            rnd_parent = chosen_parents[i]
            for l in children:
                new_relations.remove((p, l))
                new_relations.append((rnd_parent, l))

        new_tax = Taxonomy(new_relations, id_to_name=node2name)
        old_children_sets = [frozenset(set(c).intersection(all_leaves)) for p, c in taxonomy.children()]
        new_children_sets = [frozenset(set(c).intersection(all_leaves)) for p, c in new_tax.children()]
        assert frozenset(new_tax.leaves()) == frozenset(all_leaves)
        assert frozenset(old_children_sets) == frozenset(new_children_sets)
        return new_tax


class _TestShuffleLeafSets(_TestMetric, abc.ABC):
    @pytest.mark.robustness
    def test_shuffled_leaf_sets(self):
        node2name, names, ids = get_node2name(dummy_tax)
        name2node = dict(zip(names, ids))
        taxonomy = Taxonomy([(name2node[p], name2node[c]) for p, c in dummy_tax], id_to_name=node2name)

        normal_tax_score = self.metric.calculate(taxonomy.relations, node2name)
        rnd = Random(123)

        valid_score = []
        scores = []
        for _ in tqdm(range(10)):
            shuffled_taxonomy = self.shuffle_leaf_sets(rnd, taxonomy, node2name)
            shuffled_tax_score = self.metric.calculate(shuffled_taxonomy.relations, node2name)

            scores.append(shuffled_tax_score)
            if self.maximize:
                valid_score.append(int(normal_tax_score > shuffled_tax_score))
            else:
                valid_score.append(int(normal_tax_score < shuffled_tax_score))

        print("Shuffled tax score {} ({},{})".format(np.mean(scores), np.min(scores), np.max(scores)))
        print("Normal tax score {}".format(normal_tax_score))

        true_rate = sum(valid_score) / len(valid_score)
        self.assertGreater(true_rate, 0.9)


class TestRobustnessEDSC(_TestShuffleLeafSets):
    def setUp(self):
        self.metric = CscMetric(fasttext_model=self.fasttext)


class TestRobustnessSP(_TestMetric):
    def setUp(self):
        self.metric = SemanticProximity(fasttext_model=self.fasttext)

    def test_shuffle_leafs_no_difference(self):
        taxonomy = get_dummy_taxonomy()

        normal_tax_score = self.metric.calculate(taxonomy.relations, taxonomy.node2name)
        rnd = Random(123)

        for _ in tqdm(range(10)):
            shuffled_taxonomy = self.shuffle_leaf_sets(rnd, taxonomy, taxonomy.node2name)
            shuffled_tax_score = self.metric.calculate(shuffled_taxonomy.relations, taxonomy.node2name)

            self.assertEquals(normal_tax_score, shuffled_tax_score)


class TestRobustnessNSP(_TestShuffleLeafSets):
    def setUp(self):
        self.metric = NestedSemanticProximity(fasttext_model=self.fasttext)


class TestConcisenessHMH(unittest.TestCase):
    def setUp(self):
        self.metric = HumanMemoryHeuristic(threshold=7)

    def test_conciseness_hmh(self):
        node2name, _, _ = get_node2name(dummy_tax)
        hmh_score = self.metric.calculate(dummy_tax, node2name)

        self.assertTrue(hmh_score == 1)


del _TestMetric
del _TestShuffleLeafSets

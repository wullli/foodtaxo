import itertools
import unittest

from llm_food_taxonomy.graph.taxonomy import Taxonomy
import networkx as nx


class TestTaxonomy(unittest.TestCase):
    tax_edges = [
        ("food", "fruit"),
        ("food", "juice"),
        ("fruit", "apple"),
        ("fruit", "banana"),
        ("juice", "apple juice"),
        ("apple", "granny smith")
    ]

    def test_get_most_specific(self):
        tax = Taxonomy(self.tax_edges, id_to_name={i: n for i, n in enumerate(itertools.chain(*self.tax_edges))})
        res = tax.most_specific(["food", "fruit", "apple", "apple juice", "juice"])
        print(res)
        self.assertTrue(len(res) == 2)
        self.assertTrue("apple" in res and "apple juice" in res)

    def test_get_most_general(self):
        tax = Taxonomy(self.tax_edges, id_to_name={i: n for i, n in enumerate(itertools.chain(*self.tax_edges))})
        res = tax.most_general(["fruit", "apple", "apple juice", "juice"])
        print(res)
        self.assertTrue(len(res) == 2)
        self.assertTrue("fruit" in res and "juice" in res)

    def test_get_descendants(self):
        tax = Taxonomy(self.tax_edges, id_to_name={i: n for i, n in enumerate(itertools.chain(*self.tax_edges))})
        res = tax.descendants("food")
        self.assertTrue(res["food"] == set(itertools.chain(*self.tax_edges)).difference({"food"}))
        self.assertTrue(res["fruit"] == {"apple", "banana", "granny smith"})
        self.assertTrue(res["apple"] == {"granny smith"})

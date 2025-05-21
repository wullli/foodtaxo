import unittest
from functools import partial
from pathlib import Path

from llm_food_taxonomy.data.loader import load_taxonomy


class TestDataMeta(type):
    def __new__(mcs, name, bases, class_body_methods_dict):

        def _generate_splits_match_test(tax_dir, lower=False):
            def _concepts_splits_match(self):
                terms_files = list(tax_dir.glob("*.terms"))
                if len(terms_files) > 0:
                    terms, _ = load_taxonomy(str(tax_dir), with_split=True)
                    new_concepts = list(tax_dir.glob("*.new_concepts"))
                    assert len(new_concepts) == 3
                    concepts_names = {}
                    for new_concept in new_concepts:
                        split = new_concept.name.split(".")[-2]
                        concepts_names[split] = _load_concepts(new_concept, lower=lower)

                    for split, new_concepts in concepts_names.items():
                        if split == "validation":
                            split = "val"
                        split_terms = terms[terms.split == split].node_name.values.tolist()
                        for new_concept in new_concepts:
                            self.assertIn(new_concept, split_terms)

            return _concepts_splits_match

        def _generate_no_duplicates_test(tax_dir):
            def _no_duplicates(self):
                terms_files = list(tax_dir.glob("*.terms"))
                if len(terms_files) > 0:
                    terms, _ = load_taxonomy(str(tax_dir), with_split=True)
                    val = terms[terms.split == "val"].node_id.values.tolist()
                    test = terms[terms.split == "test"].node_id.values.tolist()
                    train = terms[terms.split == "train"].node_id.values.tolist()

                    self.assertTrue(len(set(val).intersection(test)) == 0)
                    self.assertTrue(len(set(val).intersection(train)) == 0)
                    self.assertTrue(len(set(test).intersection(train)) == 0)

            return _no_duplicates

        def _generate_no_duplicates_name_test(tax_dir):
            def _no_duplicates(self):
                terms_files = list(tax_dir.glob("*.terms"))
                if len(terms_files) > 0:
                    terms, _ = load_taxonomy(str(tax_dir), with_split=True)
                    val = terms[terms.split == "val"].node_name.values.tolist()
                    test = terms[terms.split == "test"].node_name.values.tolist()
                    train = terms[terms.split == "train"].node_name.values.tolist()

                    self.assertTrue(len(set(val).intersection(test)) == 0)
                    self.assertTrue(len(set(val).intersection(train)) == 0)
                    self.assertTrue(len(set(test).intersection(train)) == 0)

            return _no_duplicates

        def _load_concepts(file_path: Path, lower):
            data = []
            with open(file_path) as f:
                for line in f.readlines():
                    line = line.split("\t")[0].strip()
                    data.append(line.lower() if lower else line)
            return data

        paths = []
        data_dir = Path(__file__).parent.parent / "data"
        for dir in data_dir.iterdir():
            terms_files = list(dir.glob("*.terms"))
            if len(terms_files) > 0:
                paths.append(dir)
        for path in paths:
            class_body_methods_dict[f"test_splits_match_{path.name}"] = _generate_splits_match_test(tax_dir=path,
                                                                                                    lower=True)
            class_body_methods_dict[f"test_no_duplicates_{path.name}"] = _generate_no_duplicates_test(tax_dir=path)
            class_body_methods_dict[
                f"test_no_duplicates_name_{path.name}"] = _generate_no_duplicates_name_test(tax_dir=path)

        return type.__new__(mcs, name, bases, class_body_methods_dict)


class TestData(unittest.TestCase, metaclass=TestDataMeta):
    pass

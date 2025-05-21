import enum
import glob
import itertools
import json
import logging
import os
import queue
import re
import traceback
from abc import ABC
from collections import defaultdict
from functools import partial
from pathlib import Path
from random import Random
from typing import List, Tuple, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import dspy
from dspy import Prediction, Example
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPRO, COPRO, BootstrapFewShot
from llm_food_taxonomy.graph.taxonomy import Taxonomy
from llm_food_taxonomy.models.base import TaxonomyCompletionModel
from llm_food_taxonomy.models.hf_model import FastHFModel
from llm_food_taxonomy.models.openai_model import ConversationGPT3
from llm_food_taxonomy.models.retrieve import SimpleRetrieve, ContextVectorStore, InstructionEmbedding
from llm_food_taxonomy.models.signatures import GenerateParents, GenerateParentsUnsupervised, GenerateChildren, \
    DescribeTaxonomy, GermanGenerateParentsUnsupervised, GermanGenerateChildren, GenerateParentsUnsupervisedNoDesc

logger = logging.getLogger(__name__)


class FitMethod(enum.Enum):
    NONE = "none"
    ZERO_SHOT = "zero"
    FEW_SHOT = "few"
    FEW_SHOT_BASIC = "few_basic"


class TaxDescriptionModule(dspy.Module):
    def __init__(self, k=100, seed=123, reasoning_model=True):
        super().__init__()
        self.seed = seed
        self.k = k
        predict_class = dspy.ChainOfThought if not reasoning_model else partial(
            dspy.ChainOfThought,
            rationale_type=dspy.OutputField(
                prefix="<think>\n",
                desc="...\n<\\think>",
            ))
        self.description_cot = predict_class(DescribeTaxonomy,
                                             max_tokens=300 if not reasoning_model else 500)

    def forward(self, terms: List[str]):
        rng = Random()
        rng.seed(self.seed)
        return self.description_cot(concepts=[rng.choice(terms) for _ in range(self.k)]).description


class TaxCompletionModule(dspy.Module):
    def __init__(self, k_relations=20,
                 unsupervised=False,
                 unsupervised_children=False,
                 max_backtracks=3,
                 use_taxonomy_desc=True,
                 german=False,
                 reasoning_model=True):
        """
        DSPy Module for taxonomy completion
        :param k_relations: Num of relations to retrieve
        :unsupervised: Whether to activate unsupervised mode without checking if parents and children exist
        :max_backtracks: Maximum number of backtracks to try when output validation fails
        :use_taxonomy_desc: Whether to use the taxonomy description
        :german: Whether to use the German version of the model
        :reasoning_model: Whether to use the ChainOfThought
        """
        super().__init__()
        self.retrieve = SimpleRetrieve(k=k_relations)
        self.unsupervised = unsupervised
        self.unsupervised_children = unsupervised_children
        self.german = german
        self.parents_signature = GenerateParents
        self.use_taxonomy_desc = use_taxonomy_desc

        if self.unsupervised:
            self.parents_signature = GenerateParentsUnsupervised

            if not self.use_taxonomy_desc:
                self.parents_signature = GenerateParentsUnsupervisedNoDesc

            if self.german:
                self.parents_signature = GermanGenerateParentsUnsupervised
        predict_class = dspy.ChainOfThought if not reasoning_model else partial(
            dspy.ChainOfThought,
            rationale_type=dspy.OutputField(
                prefix="<think>\n",
                desc="...\n<\\think>",
            ))
        self.generate_parents = predict_class(self.parents_signature,
                                              max_tokens=300 if not reasoning_model else 500)
        self.generate_children = predict_class(GenerateChildren if not self.german else GermanGenerateChildren,
                                               max_tokens=300 if not reasoning_model else 500)

        self._tax = None
        self._tax_changed_relations = False
        self._tax_changed_triplets = False
        self._existing_relations = None
        self._existing_triplets = None
        self.special_char_re = re.compile(r'[^a-zA-Z0-9À-Ÿ, \n|.]')
        self.activate_assertions(max_backtracks=max_backtracks)
        self.fit_mode = False

    def _assert(self, condition, instruction):
        if self.unsupervised:
            return dspy.Assert(condition, instruction)
        else:
            return dspy.Suggest(condition, instruction)

    @property
    def existing_triplets(self):
        if self._tax_changed_triplets:
            self._existing_triplets = list(self._tax.triplets(existing=True))
            self._tax_changed_triplets = False
        return self._existing_triplets

    @property
    def existing_relations(self):
        if self._tax_changed_relations:
            self._existing_relations = [(self._tax.id_to_name[p], self._tax.id_to_name[c])
                                        for p, c in self._tax.relations]
            self._tax_changed_relations = False
        return self._existing_relations

    def _get_children(self, node):
        node_children = self._tax.node_children(node)
        names = []
        for c in node_children:
            names.append(f"{self._tax.id_to_name[c]}")
        return names

    @staticmethod
    def _filter_predicate(node_name):
        return (("none" not in node_name.lower())
                and (node_name != "None")
                and (node_name != "")
                and (node_name is not None))

    @staticmethod
    def _clean_triplet(triplet):
        parent, query, child = triplet
        parent = parent if parent is not None and len(parent.split(" ")) < 5 else None
        child = child if child is not None and len(child.split(" ")) < 5 else None
        return parent, query, child

    def _valid_triplet(self, triplet):
        cleaned_triplet = self._clean_triplet(triplet)
        return (cleaned_triplet[0] is not None) or (cleaned_triplet[2] is not None)

    def set_seed_taxonomy(self, tax: Taxonomy):
        self._tax = tax
        self._tax_changed_relations = True
        self._tax_changed_triplets = True

    def parse_comma_separated_string(self, string_value: str):
        string_value = self.special_char_re.sub("", string_value.split("\n")[0])
        nodes = list(set(
            filter(self._filter_predicate,
                   map(lambda x: x.lower().strip(),
                       string_value.strip().split(",")) if string_value != "" else set())))
        return nodes

    def parse_node_list(self, string_value: str, parent=True):
        """
        Parsing a comma separated list of node names generated by an LLM
        :param string_value: String value to parse
        :param parent: Whether to parse parents or children
        """
        nodes = self.parse_comma_separated_string(string_value)
        new_nodes = [n for n in nodes if n not in self._tax.name_to_id.keys()]
        nodes = [n for n in nodes if not n in new_nodes]

        if "None" in ", ".join(nodes):
            nodes = []

        if parent and len(nodes) > 1:
            nodes = [self._tax.id_to_name[nn]
                     for nn in self._tax.most_specific([self._tax.name_to_id[n] for n in nodes])]
        elif len(nodes) > 1:
            nodes = [self._tax.id_to_name[nn]
                     for nn in self._tax.most_general([self._tax.name_to_id[n] for n in nodes])]

        nodes += new_nodes

        if len(nodes) == 0:
            nodes += [None]

        return nodes

    def parse_triplets(self,
                       target_node: str,
                       parents: List[str],
                       children: List[str]) -> List[Tuple[str, str, str]]:
        """
        Parse positions or queries (triplets) from strings.
        :param target_node: The query or target node
        :param parents: String of the parent prediction
        :param children: String of the children prediction
        """
        existing_relations = [(p, c) for p, c in self.existing_relations if p != target_node and c != target_node]
        existing_triplets = [(p, q, c) for p, q, c in self.existing_triplets if q != target_node]

        parents, children = set(parents), set(children)
        used_children = set()
        positions = defaultdict(list)

        if self.unsupervised:
            parents = {list(parents)[0]}

        for parent in parents:
            added = 0
            parent = parent.strip() if parent is not None else None
            existing_children = [c for p, c in existing_relations if p == parent]
            for child in children:
                child = child.strip() if child is not None else None
                if child in existing_children:
                    positions[parent].append(child)
                    added += 1
                    used_children.add(child)
            if added == 0:
                positions[parent].append(None)

        for child in children.difference(used_children):
            positions[None].append(child)

        triplets = []
        for parent, children in positions.items():
            for child in children:
                assert parent != "None"
                assert child != "None"
                triplets.append((parent, target_node, child))

        if existing_triplets is not None:
            triplets = [t for t in triplets if t not in existing_triplets]
        return triplets

    def produce_triplets(self,
                         node_name: str,
                         parents: List[str],
                         children: List[str]) -> Prediction:
        """
        Produce triplet positions from parent and child predictions
        :node_name: name of the node
        :parents_prediction: Prediction instance for parents
        :children_prediction: Prediction instance for children
        :return: Prediction instance for triplets and inserted and removed relations
        """
        triplets = self.parse_triplets(
            parents=parents,
            children=children,
            target_node=node_name
        )

        new_relations, removed_relations, new_nodes = [], [], []
        if len(triplets) > 0:
            triplets = [self._clean_triplet(t) for t in triplets if self._valid_triplet(t)]
            if self.unsupervised:
                new_relations, removed_relations, new_nodes, triplets = self._tax.insert(triplets)
                self._tax_changed_relations = True
                self._tax_changed_triplets = True

        dspy.Suggest(len(triplets) > 0,
                     "No valid relations were produced.",
                     target_module=self.parents_signature)

        return dspy.Prediction(triplets=triplets,
                               new_relations=new_relations,
                               removed_relations=removed_relations,
                               new_nodes=new_nodes)

    def produce_parents(self, node_name: str, context: list, tax_desc: str, desc: str):
        """
        Produce parents for a node
        :param node_name: name of the node
        :param context: list of relations that were retrieved
        :param tax_desc: description of the taxonomy
        :param desc: description of the node
        """
        parents_prediction = self.generate_parents(child=node_name,
                                                   context=context,
                                                   taxonomy_description=tax_desc,
                                                   description=desc)
        parents = self.parse_node_list(self.special_char_re.sub("", parents_prediction.parents), parent=True)
        dspy.Suggest(parents_prediction.interpretation.strip() != "",
                     "The child concept interpretation cannot be empty.")

        existing_nodes = list(map(lambda x: self._tax.id_to_name[x], self._tax.g.nodes()))
        self.backoff(query=node_name, nodes=parents, existing_nodes=existing_nodes, node_type="parent")
        parents = [n for n in parents if len(str(n).split(" ")) < 5]
        dspy.Suggest(len(parents) > 0, "No valid parents were produced.")
        prompt = dspy.settings.config["lm"].history[-1]['prompt']
        return parents, parents_prediction.interpretation.strip(), parents_prediction, prompt

    def produce_children(self,
                         node_name: str,
                         leaf: Optional[bool],
                         parents: List[str],
                         interpretation: str,
                         context: List[str],
                         tax_desc: str,
                         desc: str):
        """
        Produce children for a node
        :param node_name: name of the node
        :param leaf: whether the node is leaf, None if unknown, usually known in unsupervised case
        :param parents: list of parent concepts of the node
        :param interpretation: interpretation of the node generated when producing parents
        :param context: list of relations that were retrieved
        :param tax_desc: description of the taxonomy
        :param desc: description of the node
        """
        try_adding_children = (leaf is not None and not leaf and self.unsupervised) or not self.unsupervised
        try:
            candidate_children = list(itertools.chain(*[self._get_children(self._tax.name_to_id[p])
                                                        for p in parents if p in self._tax.name_to_id]))
        except (KeyError, nx.NetworkXError):
            candidate_children = []

        def get_annotated_rel(rel) -> str:
            concepts = rel.split(",")
            annotated_concepts = []
            for c in concepts:
                leaf = self._tax.is_leaf(self._tax.name_to_id[c.strip()])
                leaf_annotation = "Leaf" if leaf else "Non-Leaf"
                annotated_concepts.append(f"{c.strip()} ({leaf_annotation})")
            return ", ".join(annotated_concepts)

        pred_children = []
        children_prediction = Prediction(children="")
        if len(candidate_children) > 0 and try_adding_children:
            new_ctx = [get_annotated_rel(rel) for rel in context]
            children_prediction = self.generate_children(context=new_ctx,
                                                         parent=node_name,
                                                         interpretation=interpretation,
                                                         description=desc,
                                                         taxonomy_description=tax_desc,
                                                         candidates=', '.join(candidate_children))
            if "No" in children_prediction.leaf.strip():
                pred_children = self.parse_node_list(self.special_char_re.sub("", children_prediction.children),
                                                     parent=False)
                bad = [c for c in pred_children if c not in candidate_children if c is not None]
                dspy.Suggest(len(bad) <= 0, f"{', '.join(bad)} "
                                            f"are not valid children, since they are not in the candidates.")
                try:
                    assert len(bad) == 0
                except AssertionError:
                    pred_children = []
        if None not in pred_children:
            self.backoff(query=node_name, nodes=pred_children, existing_nodes=candidate_children, node_type="child")
        pred_children = [n for n in pred_children if len(str(n).split(" ")) < 5]
        dspy.Suggest(len(pred_children) > 0, "No valid children were produced.")
        prompt = dspy.settings.config["lm"].history[-1]['prompt']
        return pred_children, children_prediction, prompt

    def backoff(self, query, nodes, existing_nodes, node_type="child") -> None:
        """
        Assert that outputs are valid and backoff if necessary
        :param query: The query node
        :param nodes: The predicted nodes to check
        :param existing_nodes: Set of nodes that the predicted nodes should come from
        :param node_type: The type of the nodes to check
        """
        for node in nodes:
            self._assert(node != query,
                         f"New node cannot be its own {node_type}.")

            if not self.unsupervised or node_type == "child":
                dspy.Suggest(node in existing_nodes,
                             f"{node} is not a valid {node_type}.")

            self._assert(len(node.split(" ")) < 5 if node is not None else True,
                         f"{node_type[0].upper}{node_type[1:]} concept has too many words: {node}")

    def forward(self, node_name, tax_desc: str, desc: str, leaf=None, return_output=False):
        """
        Forward pass for a node
        :param node_name: name of the node
        :param leaf: whether the node is leaf, None if unknown, usually known in unsupervised case
        :param tax_desc: description of the taxonomy
        :param desc: description of the node
        :param return_output: Return prediction instance for triplets and inserted and removed relations
        :return: Prediction instance for triplets and inserted and removed relations
        """
        retrieved = self.retrieve(node_name)
        context = retrieved.names
        descriptions = retrieved.descriptions
        parents, interpretation, parents_output, parent_prompt = self.produce_parents(node_name,
                                                                                      context,
                                                                                      tax_desc, desc)
        child_prompt = None
        children = []

        if not self.unsupervised or (self.unsupervised_children and self.unsupervised):
            children, _, child_prompt = self.produce_children(node_name,
                                                              leaf,
                                                              parents,
                                                              interpretation,
                                                              context,
                                                              tax_desc,
                                                              desc)

        prediction = self.produce_triplets(node_name, parents, children)
        if return_output:
            return prediction, parent_prompt, child_prompt
        else:
            return prediction


class LlmTaxonomyCompletionModel(TaxonomyCompletionModel, ABC):

    def __init__(self,
                 embedding_fn: InstructionEmbedding,
                 n_train_samples=None,
                 n_val_samples=None,
                 k_relations=5,
                 model='gpt-3.5-turbo',
                 german=False,
                 balanced=False,
                 seed=567,
                 unsupervised=False,
                 unsupervised_children=False,
                 recursive=False,
                 local_model_path="/home/support/llm",
                 max_backtracks=3,
                 quantize=0,
                 generate_taxonomy_desc=True,
                 method: FitMethod = FitMethod.NONE,
                 optimizer_options: dict = None,
                 reasoning_model=True):
        """
        Taxonomy completion using LLMs
        """
        self.input_keys = ["node_name", "desc"]
        self.k_relations = k_relations
        self.generate_taxonomy_desc = generate_taxonomy_desc
        self.balanced = balanced
        self.local_model_path = local_model_path
        self.unsupervised = unsupervised
        self.recursive = recursive
        self.fit_method = method
        self.german = german
        self.unsupervised_children = unsupervised_children
        self.optimizer_options = optimizer_options if optimizer_options is not None else {}
        self.reasoning_model = reasoning_model
        if "gpt" in model:

            self.llm = ConversationGPT3(model=model,
                                        api_key=os.environ["OPENAI_API_KEY"],
                                        top_p=0,
                                        temperature=0.0,
                                        max_tokens=500)
        else:
            model_path = None
            available_models = glob.glob(self.local_model_path)
            if model.split("/")[-1] in available_models:
                model_path = Path(self.local_model_path) / model.split("/")[-1]
            self.llm = FastHFModel(model=model,
                                   checkpoint_path=model_path,
                                   do_sample=False,
                                   temperature=0.01,
                                   max_new_tokens=500,
                                   hf_device_map="auto",
                                   max_len=4096,
                                   quantize=quantize)

        self.seed_taxonomy = None
        self.max_backtracks = max_backtracks
        self.program = TaxCompletionModule(k_relations=self.k_relations,
                                           unsupervised=self.unsupervised,
                                           unsupervised_children=self.unsupervised_children,
                                           max_backtracks=max_backtracks,
                                           german=self.german,
                                           reasoning_model=self.reasoning_model)
        self.descriptor = TaxDescriptionModule(k=100, seed=123)
        self.context_store = ContextVectorStore(embedding_fn)
        self.val_size = n_val_samples
        self.train_size = n_train_samples
        self.seed = seed
        self.tax = None
        super().__init__(config={
            "n_train_samples": n_train_samples,
            "n_val_samples": n_val_samples,
            "k_relations": self.k_relations,
            "max_backtracks": self.max_backtracks,
            "unsupervised": self.unsupervised,
            "model": model,
            "quantize": quantize,
            "method": method.value,
            "use_taxonomy_desc": self.generate_taxonomy_desc,
            "reasoning_model": self.reasoning_model,
            "unsupervised_children": self.unsupervised_children,
            **self.optimizer_options
        })

    @classmethod
    def get_natural_positions(cls, positions):
        parents = list(set([p for p, _ in positions]))
        children = list(set([c for _, c in positions]))
        parents = f"{', '.join(filter(lambda x: x is not None, parents))}"
        children = f"{', '.join(filter(lambda x: x is not None, children))}"
        return parents, children

    @staticmethod
    def get_node_list(nodes):
        parents = [p.strip() for p in nodes.split(",")]
        new_nodes = []
        for parent in parents:
            if parent is None or parent == "":
                new_nodes.append("None")
            else:
                new_nodes.append(f"{parent}")
        return ", ".join(new_nodes)

    @staticmethod
    def f1_triplets(example, pred, *_, **__):
        try:
            true_triplets = example.triplets
            pred_triplets = pred.triplets

            tp = 0
            fp = 0
            fn = 0

            for t in set(itertools.chain(*[true_triplets, pred_triplets])):
                if t in pred_triplets and t in true_triplets:
                    tp += 1
                elif t in pred_triplets and not t in true_triplets:
                    fp += 1
                elif t not in pred_triplets and t in true_triplets:
                    fn += 1
                else:
                    raise ValueError("This should never happen")

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating F1: {e}")
            f1 = 0.0
        return f1

    def _gen_triplets(self, row):
        parents = self.program.parse_comma_separated_string(row.answer[0])
        children = self.program.parse_comma_separated_string(row.answer[1])
        return self.program.parse_triplets(
            target_node=row.node_name,
            parents=parents,
            children=children,
        )

    def _create_fields(self, terms):
        terms["answer"] = terms.apply(lambda x: self.get_natural_positions(x.positions), axis=1)
        terms["parents"] = terms.apply(lambda x:
                                       self.get_node_list(x.answer[0]),
                                       axis=1)
        terms["children"] = terms.apply(lambda x:
                                        self.get_node_list(x.answer[1]),
                                        axis=1)

        terms["triplets"] = terms.apply(self._gen_triplets, axis=1)
        terms["leaf"] = terms.leaf.apply(lambda x: "Yes" if x else "No")
        return terms

    def _get_subsets(self, samples, terms, subset_size, evaluating, seed_offset=0):
        leaves = terms[terms.positions.apply(lambda x: len(x) > 0 and x[0][1] is None)].node_name.values
        nonleaves = terms[terms.positions.apply(lambda x: len(x) > 0 and x[0][1] is not None)].node_name.values

        if not evaluating:
            leave_ratio = len(leaves) / (len(leaves) + len(nonleaves))

            if subset_size is not None:
                n_leaves = int(subset_size * leave_ratio) if not self.balanced else subset_size // 2
                n_nonleaves = subset_size - n_leaves
                leaves = np.random.RandomState(self.seed + seed_offset + 1).choice(
                    leaves,
                    n_leaves,
                    replace=False)
                nonleaves = np.random.RandomState(self.seed + seed_offset + 2).choice(
                    nonleaves,
                    n_nonleaves,
                    replace=False)

                samples = [e for idx, e in enumerate(samples)
                           if e.node_name in leaves or e.node_name in nonleaves]
        return samples

    def _prepare_data(self,
                      train_terms: pd.DataFrame,
                      train_taxo: pd.DataFrame,
                      val_terms: pd.DataFrame,
                      id_to_name: dict,
                      id_to_desc: dict,
                      evaluating=False):
        dspy.configure(lm=self.llm, trace=[])
        if train_taxo is not None:
            seed_taxonomy = train_taxo.values.tolist()
            seed_taxonomy = [tuple(r) for r in seed_taxonomy]
            self.seed_taxonomy = seed_taxonomy
        else:
            self.seed_taxonomy = []
            train_terms = pd.DataFrame({"node_id": [], "node_name": [], "desc": []})

        self.tax = Taxonomy(relations=self.seed_taxonomy, id_to_name=id_to_name)
        self.program.set_seed_taxonomy(self.tax)

        if not self.recursive:
            train_terms = self._create_fields(train_terms)
            val_terms = self._create_fields(val_terms)
            if not evaluating:
                tax_desc = self.descriptor(terms=train_terms.node_name.tolist())
                train_terms["tax_desc"] = tax_desc
                val_terms["tax_desc"] = tax_desc
            train_terms.drop(columns=["leaf"], inplace=True)
            val_terms.drop(columns=["leaf"], inplace=True)
        else:
            val_terms["leaf"] = [True] * len(val_terms)

        input_keys = ["node_name", "tax_desc", "desc"]
        train_samples = None
        if not self.recursive:
            train_samples = [dspy.Example({str(c): row[str(c)] for c in train_terms.columns}).with_inputs(*input_keys)
                             for _, row in train_terms.iterrows()]
        val_samples = [dspy.Example({str(c): row[str(c)] for c in val_terms.columns}).with_inputs(*input_keys)
                       for _, row in val_terms.iterrows()]

        if not self.recursive:
            train_samples = self._get_subsets(train_samples, train_terms, subset_size=self.train_size,
                                              evaluating=evaluating)
            val_samples = self._get_subsets(val_samples, val_terms, subset_size=self.val_size,
                                            evaluating=evaluating,
                                            seed_offset=2)
        self.context_store.fill(seed_taxonomy=self.seed_taxonomy,
                                train_samples=train_samples,
                                id_to_name=id_to_name,
                                id_to_desc=id_to_desc,
                                recursive=self.recursive,
                                evaluating=evaluating)
        dspy.configure(lm=self.llm, rm=self.context_store.retrieval_model, trace=[])
        return train_samples, val_samples

    def fit(self,
            train_terms: pd.DataFrame,
            train_taxo: pd.DataFrame,
            val_terms: pd.DataFrame,
            id_to_name: dict = None,
            id_to_desc: dict = None):
        """
        Expand the taxonomy with the given leaves
        :param train_terms: The train/seed nodes/terms
        :param train_taxo: The train/seed taxonomy
        :param val_terms: The new concepts to expand the taxonomy with
        :param id_to_name: The mapping from node id to node name
        """
        self.program.fit_mode = True
        train_samples, val_samples = self._prepare_data(train_terms, train_taxo, val_terms, id_to_name, id_to_desc)
        evaluate = Evaluate(devset=val_samples, metric=self.f1_triplets, display_progress=True)

        self.program = TaxCompletionModule(k_relations=self.k_relations,
                                           unsupervised=self.unsupervised,
                                           unsupervised_children=self.unsupervised_children,
                                           max_backtracks=self.max_backtracks,
                                           german=self.german,
                                           reasoning_model=self.reasoning_model)
        self.program.set_seed_taxonomy(self.tax)

        baseline_train_score = evaluate(self.program, devset=train_samples, display_progress=True)
        baseline_eval_score = evaluate(self.program, devset=val_samples, display_progress=True)
        eval_kwargs = dict(num_threads=10, display_progress=True, display_table=0)

        print(f"Fit method: {self.fit_method}")

        if self.fit_method.value == FitMethod.ZERO_SHOT.value:
            print("Fitting with COPRO")
            breadth = self.optimizer_options.pop('breadth', 10)
            depth = self.optimizer_options.pop('depth', 10)
            opt = COPRO(metric=self.f1_triplets, breadth=breadth, depth=depth)
            self.program = opt.compile(self.program,
                                       trainset=train_samples,
                                       **self.optimizer_options,
                                       eval_kwargs=eval_kwargs)
        elif self.fit_method.value == FitMethod.FEW_SHOT.value:
            print("Fitting with MIPRO")
            num_candidates = self.optimizer_options.pop('num_candidates', 10)
            view_data_batch_size = self.optimizer_options.pop('view_data_batch_size', 5)
            opt = MIPRO(metric=self.f1_triplets, num_candidates=num_candidates,
                        view_data_batch_size=view_data_batch_size)
            self.program = opt.compile(self.program,
                                       trainset=train_samples,
                                       requires_permission_to_run=False,
                                       **self.optimizer_options,
                                       eval_kwargs=eval_kwargs)
        elif self.fit_method.value == FitMethod.FEW_SHOT_BASIC.value:
            print("Fitting with BootstrapFewShot")
            opt = BootstrapFewShot(metric=self.f1_triplets, **self.optimizer_options)
            self.program = opt.compile(self.program,
                                       trainset=train_samples)

        train_score = evaluate(self.program, devset=train_samples, display_progress=True)
        eval_score = evaluate(self.program, devset=val_samples, display_progress=True)
        self.program.fit_mode = False

        return {"train_score": train_score,
                "eval_score": eval_score,
                "baseline_train_score": baseline_train_score,
                "baseline_eval_score": baseline_eval_score}

    @staticmethod
    def _stream_output(stream_dir: str,
                       query_node: str,
                       new_position: list,
                       new_nodes: dict) -> None:
        pred_file = (stream_dir / "predictions.pos.json")
        terms_file = (stream_dir / "concepts.terms")
        pred_file.touch(exist_ok=True)
        terms_file.touch(exist_ok=True)
        with open(pred_file, "a") as f:
            f.write(f"{json.dumps({query_node: new_position})}\n")
        with open(terms_file, "a") as f:
            for id, name in new_nodes.items():
                f.write(f"{name}\t{name}\n")

    def complete(self,
                 test_terms: pd.DataFrame,
                 train_terms: pd.DataFrame = None,
                 train_taxo: pd.DataFrame = None,
                 id_to_name: dict = None,
                 id_to_desc: dict = None,
                 return_output=False,
                 depth=10,
                 stream_dir=None
                 ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Expand the taxonomy with the given leaves
        :param train_terms: The train/seed nodes/terms
        :param train_taxo: The train/seed taxonomy
        :param test_terms: The nodes to attach
        :param id_to_name: The id to start with for the newly generated nodes
        :param id_to_desc: The description of the nodes
        :param return_output: Return the generated text
        :param depth: The maximum depth of a node away from a leaf to qualify for prediction
        :return: The expanded taxonomy
        """
        logger.debug("Preparing data..")
        _, test_samples = self._prepare_data(train_terms, train_taxo, test_terms, id_to_name, id_to_desc,
                                             evaluating=True)

        positions = {}
        outputs = {}

        if self.generate_taxonomy_desc:
            tax_desc = self.descriptor(terms=test_terms.node_name.tolist())
        else:
            tax_desc = None

        q = queue.Queue()
        for ts in test_samples:
            q.put((ts, 0))

        expanded = 0
        with tqdm(desc="Generating predictions...", postfix={"remaining": q.qsize()}) as pb:
            while not q.empty():
                s, level = q.get()
                try:
                    pred = self.program(s.node_name,
                                        leaf=None if not hasattr(s, "leaf") else s.leaf,
                                        desc=s.desc,
                                        tax_desc=tax_desc,
                                        return_output=return_output)
                    if return_output:
                        pred, parents, children = pred
                        outputs[s.node_name] = {}
                        outputs[s.node_name] = {
                            "parents": parents,
                            "children": children
                        }
                    if len(pred.triplets) == 0:
                        print("No triplets were generated for node", s.node_name)

                    pos = [(t[0], t[2]) for t in pred.triplets]
                    if s.node_name not in positions.keys():
                        positions[s.node_name] = pos
                    else:
                        positions[s.node_name].extend(pos)

                    if self.unsupervised and self.recursive:
                        for n in pred.new_nodes:
                            # new_level = level + 1
                            # if (depth - 1) >= new_level:
                            q.put((Example(node_id=n[0], node_name=n[1], leaf=False), 0))
                        self.context_store.add_new_docs(pred=pred, id_to_name=id_to_name, id_to_desc=id_to_desc)
                        if expanded > len(test_samples) and len(self.tax.roots()) == 1:
                            break

                        if stream_dir is not None:
                            self._stream_output(stream_dir=stream_dir,
                                                query_node=s.node_name,
                                                new_position=positions[s.node_name],
                                                new_nodes={self.tax.name_to_id[n]: n
                                                           for _, n in [(None, s.node_name)] + pred.new_nodes})
                except Exception as e:
                    logging.error(f"Error processing node: {s.node_name} due to '{e}'")
                    traceback.print_exc()
                pb.update(1)
                pb.set_postfix({"remaining": q.qsize()})
                expanded += 1
        if not return_output:
            return positions
        else:
            return positions, outputs

    def save(self, path: str):
        self.program.save(path)

    def load(self, path: str):
        self.program = TaxCompletionModule(k_relations=self.k_relations,
                                           unsupervised=self.unsupervised,
                                           unsupervised_children=self.unsupervised_children,
                                           max_backtracks=self.max_backtracks,
                                           german=self.german,
                                           reasoning_model=self.reasoning_model)
        self.program.load(path)

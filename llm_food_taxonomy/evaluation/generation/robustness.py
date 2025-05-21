import abc
from itertools import product
from typing import List, Tuple, Dict, Hashable, Generator, Union, Set

import networkx as nx
import numpy as np
from scipy.stats import kendalltau

try:
    import cupy as cp

    print("Using CuPy backend.")
except ImportError:
    cp = np
    print("CuPy not installed.")

import pandas as pd
import spacy
import torch
from datasets import Dataset
from collections import defaultdict

from nltk import WordNetLemmatizer
from scipy.spatial.distance import cosine
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_2d
from tqdm.auto import tqdm
from transformers import pipeline, Pipeline
from transformers.pipelines.base import KeyDataset

from llm_food_taxonomy.evaluation.metric import UnsupervisedMetric
from llm_food_taxonomy.graph.taxonomy import Taxonomy


def cosine_distance(vec_a, vec_b) -> np.ndarray:
    return cosine(vec_a, vec_b)


def cosine_similarity(vec_a, vec_b) -> np.ndarray:
    return 1 - cosine_distance(vec_a, vec_b)


class SentenceTransformerUnsupervisedMetric(UnsupervisedMetric, abc.ABC):
    def __init__(self,
                 use_descriptions=False,
                 sentence_transformer=None,
                 progress=False,
                 batch_size=128) -> None:
        self.progress = progress
        self.batch_size = batch_size
        self.use_descriptions = use_descriptions
        if sentence_transformer is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').eval()
        else:
            self.model = sentence_transformer
        self.model.eval()

    def embeddings(self, node2name: dict, descriptions: dict = None):
        with torch.no_grad():
            if self.use_descriptions:
                sentence_map = {
                    nid: f"{nn}: {descriptions.get(nid, nn)}" if isinstance(descriptions.get(nid, nn), str) else nn
                    for nid, nn in node2name.items()
                }
            else:
                sentence_map = {nid: nn for nid, nn in node2name.items()}

        sentences = list(sentence_map.values())
        embeddings = self.model.encode(sentences, show_progress_bar=True, batch_size=32)
        embeddings = dict(zip(sentence_map.keys(), embeddings))
        return embeddings

    def similarity_map(self, emb: dict):
        with torch.no_grad():
            keys = np.array(list(emb.keys()))
            emb_tensor = torch.tensor(np.array(list(emb.values()))).to(self.model.device)
            similarity_map = {}

            def _batch_cosine_similarity(emb_batch: torch.Tensor, key_batch: torch.Tensor):
                if len(emb_batch) == 0:
                    return
                sim_mat = util.cos_sim(emb_batch, emb_tensor).cpu().numpy()
                for i, c1 in enumerate(key_batch):
                    for j, c2 in enumerate(emb.keys()):
                        similarity_map[frozenset((c1, c2))] = sim_mat[i, j]

            end_idx = 0
            for i in tqdm(range(0, len(emb) // self.batch_size), desc="Precomputing Similarity"):
                start_idx, end_idx = i * self.batch_size, min((i * self.batch_size) + self.batch_size, len(emb) - 1)
                emb_batch = emb_tensor[start_idx:end_idx]
                key_batch = keys[start_idx:end_idx]
                assert len(emb_batch) == len(key_batch) == self.batch_size, \
                    f"{len(emb_batch), len(key_batch), self.batch_size}"
                _batch_cosine_similarity(emb_batch, key_batch)

            overflow_batch = emb_tensor[end_idx:]
            overflow_keys = keys[end_idx:]
            if len(overflow_batch) > 0:
                _batch_cosine_similarity(overflow_batch, overflow_keys)
            return similarity_map


class CscMetric(SentenceTransformerUnsupervisedMetric):

    def __init__(self,
                 use_descriptions=False,
                 sentence_transformer=None,
                 progress=False) -> None:
        super().__init__(use_descriptions=use_descriptions, sentence_transformer=sentence_transformer,
                         progress=progress)

    @staticmethod
    def _wu_palmer_similarity(path1, path2):
        """
        Compute Wu-Palmer similarity given two paths from the root to the concepts.

        :param path1: List of nodes from the root to concept1
        :param path2: List of nodes from the root to concept2
        :return: Wu-Palmer similarity score
        """
        lcs_depth = 0
        for a, b in zip(path1, path2):
            if a == b:
                lcs_depth += 1
            else:
                break

        depth1, depth2 = len(path1), len(path2)
        return (2 * lcs_depth) / (depth1 + depth2) if (depth1 + depth2) > 0 else 0

    @classmethod
    def wu_palmer_similarity(cls, concept1, concept2, ancestries) -> float:
        max_score = float('-inf')

        for a1 in ancestries.get(concept1, [()]):
            for a2 in ancestries.get(concept2, [()]):
                max_score = max(max_score, cls._wu_palmer_similarity(a1, a2))

        return max_score

    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  subset: Set[str] = None,
                  similarity_map: dict = None) -> float:
        tax = Taxonomy(taxonomy_relations, node2name)
        tax.connect()

        path_dists = []
        emb_dists = []
        it = tax.children()
        n_nodes = len(it)
        centroids = {}

        # Siblings should be more similar than parent-child
        ancestries = defaultdict(list)
        for a in tax.ancestries():
            ancestries[a[-1]].append(a)

        if similarity_map is None:
            embeddings = self.embeddings(node2name, descriptions)
            for p, cs in it:
                if subset is not None and p not in subset:
                    continue
                centroids[p] = embeddings[p]
            similarity_map = self.similarity_map(centroids)

        nodes = tax.g.nodes()
        it = product(nodes, nodes)
        if self.progress:
            it = tqdm(it, total=n_nodes ** 2, desc="Calculating CSC", mininterval=1)
        for c1, c2 in it:
            if c1 == c2 or frozenset((c1, c2)) not in similarity_map:
                continue
            path_dists.append(self.wu_palmer_similarity(c1, c2, ancestries))
            emb_dists.append(similarity_map[frozenset((c1, c2))])

        return {"csc_coef": kendalltau(path_dists, emb_dists).statistic}


class SemanticProximity(SentenceTransformerUnsupervisedMetric):

    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  subset: Set[str] = None,
                  similarity_map: dict = None) -> float:
        tax = Taxonomy(taxonomy_relations, node2name)
        tax.connect()

        children = tax.children()
        leaves = set(tax.leaves())

        if similarity_map is None:
            embeddings = self.embeddings(node2name, descriptions)
            similarity_map = self.similarity_map({k: v for k, v in embeddings.items() if k in leaves})

        scores = []

        it = children
        if self.progress:
            it = tqdm(it, desc="Calculating Semantic Proximity")
        for _, cs in it:
            leaf_children = [c for c in cs if c in leaves]
            if len(leaf_children) > 1:
                min_sim = min(similarity_map[frozenset((c1, c2))]
                              for c1, c2 in product(*[leaf_children, leaf_children]) if c1 != c2)
                outsiders = leaves.difference(leaf_children)
                outside_sims = np.array([similarity_map[frozenset((c1, c2))]
                                         for c1, c2 in product(*[leaf_children, outsiders]) if
                                         c1 != c2])
                sorted_outside_sims = np.sort(outside_sims)[::-1]
                intruders = np.sum(sorted_outside_sims >= min_sim)
                s = 1 - (intruders / len(sorted_outside_sims))
                scores.append(s)
        return np.mean(scores)


class NestedSemanticProximity(SentenceTransformerUnsupervisedMetric):

    def intruders(self,
                  parent: Hashable,
                  group: set,
                  descendants: Dict[Hashable, set],
                  similarity_map: Dict[frozenset, float],
                  g: nx.DiGraph,
                  min_sim: float) -> Generator[float, None, None]:
        for ancestor in g.predecessors(parent):
            outsiders = descendants[ancestor] - group
            score = 1
            if len(outsiders) > 0:
                outsider_sim = [similarity_map[frozenset((c1, c2))]
                                for c1, c2 in product(*[group, outsiders]) if c1 != c2]
                num_intruders = np.sum(outsider_sim >= min_sim)
                score -= (num_intruders / len(outsider_sim))
            yield score
            yield from self.intruders(ancestor, group, descendants, similarity_map, g, min_sim)

    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  subset: Set[str] = None,
                  similarity_map: dict = None) -> float:
        embeddings = self.embeddings(node2name, descriptions)

        tax = Taxonomy(taxonomy_relations, node2name)
        tax.connect()
        descendants = tax.descendants()
        children = tax.children()
        leaves = set(tax.leaves())

        similarity_map = {}
        emb = np.array([embeddings[n] for n in leaves])
        sim_mat = cosine_similarity_2d(emb)
        for i, c1 in enumerate(leaves):
            for j, c2 in enumerate(leaves):
                similarity_map[frozenset((c1, c2))] = sim_mat[i, j]

        it = children
        scores = []
        if self.progress:
            it = tqdm(it, desc="Calculating Nested Semantic Proximity")
        for p, cs in it:
            leaf_children = [c for c in cs if c in leaves]
            if len(leaf_children) > 1:
                intra_group_min = min(similarity_map[frozenset((c1, c2))]
                                      for c1, c2 in product(*[leaf_children, leaf_children]) if c1 != c2)
                group_scores = list(self.intruders(p, set(leaf_children), descendants, similarity_map, tax.g,
                                                   min_sim=intra_group_min))
                if len(group_scores) > 0:
                    scores.append(np.mean(group_scores))

        return np.mean(scores)


class NliScorer:
    def __init__(self, model_name: str, progress: bool = True):
        self.nli = pipeline("text-classification", model=model_name, batch_size=128)
        self.progress = progress

    def __call__(self, queries: list[str]):
        outputs = []
        premises = []
        hypotheses = []
        for q in queries:
            premise, hypothesis = q["text"].split(".")
            premises.append(premise)
            hypotheses.append(hypothesis)
        query_ds = Dataset.from_dict({"text": premises, "text_pair": hypotheses})

        nli = self.nli(query_ds.to_list(), top_k=None)
        if self.progress:
            nli = tqdm(nli, total=len(query_ds), desc="Running Queries")

        for out in nli:
            outputs.append(out)

        return outputs

    @staticmethod
    def strong_score(r, probability=True) -> float:
        prob = [d for d in r if d["label"].lower() == "entailment"][0]["score"]
        if probability:
            return prob
        return int(prob > 0.5)

    @staticmethod
    def weak_score(r, probability=True) -> float:
        prob = 1 - [d for d in r if d["label"].lower() == "contradiction"][0]["score"]
        if probability:
            return prob
        return int(prob > 0.5)


class BartNliScorer(NliScorer):

    def __init__(self, model_name="facebook/bart-large-mnli", progress: bool = True):
        super().__init__(model_name, progress)


class ModernBertNliScorer(NliScorer):

    def __init__(self, model_name="tasksource/ModernBERT-base-nli", progress: bool = True):
        super().__init__(model_name, progress)


class RobertaNliScorer(NliScorer):

    def __init__(self, model_name="FacebookAI/roberta-large-mnli", progress: bool = True):
        super().__init__(model_name, progress)


class FeverDebertaNliScorer(NliScorer):

    def __init__(self, model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", progress: bool = True):
        super().__init__(model_name, progress)


class NliMetric(UnsupervisedMetric):
    def __init__(self,
                 scorer: NliScorer,
                 strict=False,
                 probability=True,
                 progress=False,
                 propagate=True):
        self.nlp = spacy.load("en_core_web_sm")
        self.scorer = scorer
        self.progress = progress
        self.strict = strict
        self.probability = probability
        self.cache = {}
        self.propagate = propagate

    @staticmethod
    def get_pronoun(word) -> str:
        return "an" if word[0] in ["a", "e", "i", "o", "u"] else "a"

    @staticmethod
    def iterate_edges(path):
        """
        Iterate over edges from a given path of nodes.
        Yields edges as tuples in the format (second, first).
        """
        if len(path) < 2:
            return
        for i in range(len(path) - 1):
            yield path[i], path[i + 1]

    def strong_score(self, r) -> float:
        prob = [d for d in r if d["label"].lower() == "entailment"][0]["score"]
        if self.probability:
            return prob
        return int(prob > 0.5)

    def weak_score(self, r) -> float:
        prob = 1 - [d for d in r if d["label"].lower() == "contradiction"][0]["score"]
        if self.probability:
            return prob
        return int(prob > 0.5)

    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  subset: Set[str] = None,
                  similarity_map: dict = None,
                  return_scores: bool = False) -> float:
        tax = Taxonomy(taxonomy_relations, node2name)
        if self.propagate:
            tax.connect()

        queries = []
        cached_mask = []
        for i, (p, c) in enumerate(tax.g.edges()):
            pn, cn = node2name.get(p, p), node2name.get(c, c)
            child_desc = descriptions.get(cn, cn)
            child_lemma, parent_lemma = self.nlp(cn)[0].lemma_, self.nlp(pn)[0].lemma_
            query = f"{child_desc}. {child_lemma} is a kind of {parent_lemma}"
            queries.append({"text": query})
            cached_mask.append(query in self.cache)

        cached_mask = np.array(cached_mask)
        total = len(queries)
        queries_to_run = np.array(queries)[~cached_mask].tolist()
        outputs = self.scorer(queries_to_run)

        results = []
        it = range(total)
        running_strong_sum = 0
        running_weak_sum = 0
        scores = defaultdict(list)
        for i in it:
            is_cached = cached_mask[i]
            if not is_cached:
                out = outputs[np.cumsum(~cached_mask)[i] - 1]
                results.append(out)
                self.cache[queries[i]["text"]] = out
            else:
                results.append(self.cache[queries[i]["text"]])
            strong_score = self.strong_score(results[-1])
            weak_score = self.weak_score(results[-1])
            scores["NLIV-Strong"].append(strong_score)
            scores["NLIV-Weak"].append(weak_score)
            running_strong_sum += strong_score
            running_weak_sum += weak_score

        res = {"NLIV-Strong": running_strong_sum / total, "NLIV-Weak": running_weak_sum / total}
        if self.propagate:
            out = {}
            for score_type, score_values in scores.items():
                running_sum = 0
                score_map = dict(zip(tax.g.edges(), score_values))
                ancestries = tax.ancestries()
                n_roots = len(tax.roots())
                if self.progress:
                    ancestries = tqdm(ancestries, desc="Propagating Scores")

                for a in ancestries:
                    if len(a) < 2:
                        continue
                    propagated_score = 1
                    n_edges = 0
                    for edge in self.iterate_edges(a):
                        n_edges += 1
                        propagated_score *= score_map[edge]
                    # nth root of the product of scores, geometric mean
                    propagated_score = propagated_score ** (1 / n_edges)
                    running_sum += propagated_score
                divisor = len(ancestries) - n_roots
                out[score_type] = running_sum / divisor if divisor > 0 else 0
            res = out
        if return_scores:
            return scores
        return res


class RaTEMetric(UnsupervisedMetric):

    def __init__(self,
                 pattern_path: str,
                 top_k: int,
                 model: str = 'bert-large-uncased-whole-word-masking',
                 batch_size: int = 128) -> None:
        self.pattern_path = pattern_path
        self.model = model
        self.top_k = top_k

        self.query_templates = []
        self.column_names = []
        self.cached_queries = {}
        with open(self.pattern_path, "r") as fin:
            for line in fin.readlines():
                if not line.startswith("#"):
                    query, name = line.split(",")
                    self.query_templates.append(query.strip())
                    self.column_names.append(name.strip())

        if torch.cuda.is_available():
            print("Using CUDA backend.")
            device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("Using MPS backend.")
            device = torch.device('mps')
        else:
            print("Calculating using CPU.")
            device = torch.device('cpu')

        self.unmasker = pipeline('fill-mask', model=self.model, device=device, batch_size=batch_size)
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_a_word(self, word):
        split_word = word.split()
        # print(split_word)
        ngram = len(split_word)
        if ngram == 1:
            return self.lemmatizer.lemmatize(word, pos='n')
        else:
            last_word = split_word[-1]
            lem = self.lemmatizer.lemmatize(last_word, pos='n')
            new_word = ' '.join(split_word[:-1] + [lem])
            return new_word

    def generate_queries_from_template(self, token):
        test_queries = []

        for template in self.query_templates:
            test_queries.append(template.replace("{token}", token) + " .")

        return test_queries

    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  similarity_map: dict = None) -> float:
        predictions = {}

        eval_parents = []
        eval_children = []

        for parent, child in taxonomy_relations:
            eval_parents.append(parent)
            eval_children.append(child)

        n_templates = len(self.generate_queries_from_template("dummy"))
        queries = []
        cached_mask = []
        for child in eval_children:
            child_queries = self.generate_queries_from_template(child)
            queries.append(child_queries)
            cached_mask.append(child in self.cached_queries)

        cached_mask = np.array(cached_mask)
        queries_unmasked = np.array(self.unmasker(np.array(queries)[~cached_mask].ravel().tolist(),
                                                  top_k=self.top_k)).reshape(-1, n_templates, self.top_k)

        for i, child in enumerate(eval_children):
            if not cached_mask[i]:
                unmasked_idx = np.cumsum(~cached_mask)[i] - 1
                self.cached_queries[child] = queries_unmasked[unmasked_idx]
            unmasked = self.cached_queries[child]

            for j, column in enumerate(self.column_names):
                if child not in predictions:
                    predictions[child] = {}
                predictions[child][column] = unmasked[j]

        df_result = pd.DataFrame()
        df_result["hypernym"] = eval_parents
        df_result["hyponym"] = eval_children

        for column_name in self.column_names:
            query_predictions = []

            for parent, child in zip(eval_parents, eval_children):
                preds = predictions[child][column_name]

                parent_in_preds = False

                for pred_list in preds:
                    if parent == pred_list['token_str']:
                        parent_in_preds = True
                        break
                query_predictions.append(parent_in_preds * 1)

            df_result[column_name] = query_predictions

        df_result['sum'] = df_result[self.column_names].sum(axis=1)
        return len(df_result.loc[df_result["sum"] > 0]) / len(df_result)

import abc
import random

import chromadb
import numpy as np
import torch
from chromadb import Settings
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import dsp
from dspy import Prediction, Retrieve, Example
from typing import Union, List, Optional, Tuple, Dict

from dspy.retrieve.chromadb_rm import ChromadbRM


class InstructionEmbedding(abc.ABC):

    @abc.abstractmethod
    def embed_queries(self, queries: list[str]) -> Tuple:
        ...

    @abc.abstractmethod
    def embed_passages(self, passage: list[str]) -> Tuple:
        ...


class GteEmbedding(InstructionEmbedding):

    def __init__(self, model='Alibaba-NLP/gte-large-en-v1.5', batch_size: int = 128,
                 ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True).eval()
        self.model.max_seq_length = 512
        self.batch_size = batch_size

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(queries,
                                    max_length=self.model.max_seq_length,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt')
            embeddings = self.model(**inputs).last_hidden_state[:, 0]
            return F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()

    def embed_passages(self, passages: list[str]) -> np.ndarray:
        return self.embed_queries(passages)


class SimpleRetrieve(Retrieve):

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> Prediction:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [query.strip().split('\n')[0].strip() for query in queries]

        k = k if k is not None else self.k
        passages = dsp.retrieve(queries, k=k)
        names, descriptions = [], []
        for p in passages:
            nodes = [n.strip() for n in p.split(",")]
            name_pair = [n.split(":")[0].strip() for n in nodes]
            names.append(", ".join(name_pair))
            descriptions.extend(nodes)

        return Prediction(names=names, descriptions=descriptions)


class ContextVectorStore:

    def __init__(self, embedding_function: InstructionEmbedding) -> None:
        """
        Wrapper around ChromaDB in-memory vector store for providing contextual information
        :param embedding_function: function that takes contexts as input and returns vector representation
        """
        self.embedding_function = embedding_function
        self.collection_name = f"taxonomy_relations"
        self.retrieval_model = None
        self.client = None
        self.collection = None
        self.deleted = 0

    def _get_relations_docs(self,
                            seed_taxonomy: List[Tuple[str, str]],
                            id_to_name: Dict[str, str],
                            id_to_desc: Dict[str, str]) -> (List[str], List[List[str]]):
        return list(set([self._join_document_strings(tuple([f"{id_to_name[nid]}: {id_to_desc[nid]}" for nid in r]))
                         for r in seed_taxonomy]))

    @staticmethod
    def _join_document_strings(data_elements: Tuple, sep=", ") -> str:
        return sep.join(data_elements)

    def add_new_docs(self, pred: Prediction, id_to_name: dict, id_to_desc: dict):
        """
        Adding new docs to the collection
        :param pred: prediction with new relations
        :param id_to_name: mapping of id to name
        :param id_to_desc: mapping of id to description
        """
        new_docs = list(self._get_relations_docs(pred.new_relations, id_to_name=id_to_name, id_to_desc=id_to_desc))
        if len(new_docs) > 0:
            embeddings = self.embedding_function.embed_passages(new_docs)
            self.collection.add(ids=[str(id + self.collection.count() + self.deleted) for id in range(len(new_docs))],
                                embeddings=embeddings,
                                documents=new_docs)
        old_docs = list(self._get_relations_docs(pred.removed_relations, id_to_name=id_to_name, id_to_desc=id_to_desc))
        if len(old_docs) > 0:
            where = {"$or": [{"$contains": d} for d in old_docs]} if len(old_docs) > 1 else {"$contains": old_docs[0]}
            self.collection.delete(where_document=where)
            self.deleted += len(old_docs)

    def fill(self,
             seed_taxonomy: List[Tuple[str, str]],
             train_samples: List[Example],
             id_to_name: dict,
             id_to_desc: dict,
             recursive: bool,
             evaluating=False):
        """
        Fill the vector store with the context relations
        :param seed_taxonomy: seed taxonomy, a list of tuples (id parent, id child)
        :param train_samples: list of training examples
        :param id_to_name: mapping of id to name
        :param id_to_desc: mapping of id to description
        :param recursive: whether to use the unsupervised method or not
        :param evaluating: whether we are in evaluation mode or not
        """
        self.deleted = 0
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.client.reset()
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        docs = list(self._get_relations_docs(seed_taxonomy, id_to_name=id_to_name, id_to_desc=id_to_desc))

        if not recursive and not evaluating:
            train_names = [example.node_name for example in train_samples]
            docs = [d for d in docs if all(n.strip() not in train_names for n in d.split(","))]

        self.retrieval_model = ChromadbRM(self.collection_name,
                                          persist_directory=self.client.get_settings().persist_directory,
                                          embedding_function=self.embedding_function.embed_queries,
                                          client=self.client)
        if len(docs) > 0:
            embeddings = self.embedding_function.embed_passages(docs)
            self.collection.add(ids=[str(id) for id in range(len(docs))],
                                embeddings=embeddings,
                                documents=docs)

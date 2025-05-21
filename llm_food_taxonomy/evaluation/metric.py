from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Set, List, Tuple, Dict

import numpy as np


@dataclass
class ScoreAccumulator:
    tp: float = 0
    fp: float = 0
    fn: float = 0
    tn: float = 0

    @property
    def total(self):
        if 0 in [self.tp, self.fp, self.fn, self.tn]:
            raise ValueError("One of the accumulator values is 0, total count is possibly bad!")
        return self.tp + self.fp + self.fn + self.tn

    def precision(self, nan=False):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0.0 else (0.0 if not nan else np.nan)

    def recall(self, nan=False):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0.0 else (0.0 if not nan else np.nan)

    def f1(self, nan=False):
        p = self.precision(nan=nan)
        r = self.recall(nan=nan)
        denom = (p + r)
        return 2 * p * r / denom if (denom > 0.0 and not np.isnan(denom)) else (0.0 if not nan else np.nan)

    def accuracy(self):
        return (self.tp + self.tn) / self.total


class Metric:
    @classmethod
    @abstractmethod
    def calculate(cls,
                  pred_positions: Dict[str, List[Tuple[str, str]]],
                  true_positions: Dict[str, List[Tuple[str, str]]],
                  node2name: Dict[str, str],
                  seed_taxonomy: List[Tuple[str, str]],
                  leaves: Set[str] = None) -> dict:
        raise NotImplementedError("Metric is not implemented!")


class UnsupervisedMetric:
    @abstractmethod
    def calculate(self,
                  taxonomy_relations: List[Tuple[str, str]],
                  node2name: Dict[str, str],
                  descriptions: Dict[str, str] = None,
                  similarity_map: dict = None) -> float:
        raise NotImplementedError("Metric is not implemented!")

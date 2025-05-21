import logging
from abc import abstractmethod
from typing import Iterable, List, Tuple

import pandas as pd


class TaxonomyCompletionModel:

    def __init__(self, config: dict):
        """
        Initialize the model
        :param config: The configuration of the model, e.g. hyperparameters
        """
        self._config = config

    @abstractmethod
    def fit(self,
            train_terms: pd.DataFrame,
            train_taxo: pd.DataFrame,
            val_terms: pd.DataFrame,
            id_to_name: dict = None):
        """
        Expand the taxonomy with the given leaves
        :param train_terms: The train/seed nodes/terms
        :param train_taxo: The train/seed taxonomy
        :param val_terms: The new concepts to expand the taxonomy with truth positions
        :param id_to_name: The mapping from node id to node name
        """
        raise NotImplementedError("Fit method not implemented")

    @abstractmethod
    def complete(self,
                 train_terms: pd.DataFrame,
                 train_taxo: pd.DataFrame,
                 test_terms: pd.DataFrame,
                 id_to_name: dict = None,
                 **kwargs) -> List[Tuple[str, str, str]]:
        """
        Expand the taxonomy with the given leaves
        :param train_terms: The train/seed nodes/terms
        :param train_taxo: The train/seed taxonomy
        :param test_terms: The new concepts to expand the taxonomy with
        :param id_to_name: The id to start with for the newly generated nodes
        :return: The expanded taxonomy
        """
        raise NotImplementedError("Expand method not implemented")

    @property
    def config(self):
        """
        Get the configuration of the model, e.g. hyperparameters
        :return:
        """
        return self._config

    @staticmethod
    def disable_loggers(exclude=tuple()):
        """
        Disable all loggers except for the logger for this class
        :return:
        """
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for log in loggers:
            if log.name not in exclude:
                log.setLevel(logging.WARNING)

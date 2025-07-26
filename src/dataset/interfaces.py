"""
Interface module for dataset loading strategies.

This module defines an abstract base class intended to be used
as a standard interface for implementing dataset loading logic.
It follows the Strategy design pattern, allowing the replacement
of different data loading mechanisms without modifying the consumer code.

Use cases include modular pipelines and reusable components
across multiple data sources such as files, APIs, databases, or synthetic generators.
"""

from abc import ABC, abstractmethod
from typing import Any

class IDatasetLoaderStrategy(ABC):
    """
    Base interface for dataset loading strategies.

    Any class implementing this interface must provide
    a concrete implementation of the `load` method,
    which is responsible for retrieving raw data for processing.

    Examples of concrete implementations include:
    - Reading local CSV files
    - Accessing relational databases
    - Querying external APIs
    - Generating synthetic data
    """

    @abstractmethod
    def load(self) -> Any:
        """
        Loads the dataset.

        Returns
        -------
        Any
            The return type may vary depending on the implementation,
            but it is typically a pandas.DataFrame or a compatible structure.
        """
        pass

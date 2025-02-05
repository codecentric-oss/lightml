"""Module for dataset"""

from abc import abstractmethod

from abc import ABC
from enum import Enum

import pandas as pd
from copy import deepcopy

from canaryml.datadescriptions import DataDescription


class MissingSubsetError(Exception):
    def __init__(self, message: str = "The dataset has no subset information."):
        self.message = message


class Subset(str, Enum):
    TEST: str = "test"
    TRAIN: str = "train"
    VALIDATION: str = "validation"


class Dataset(ABC):
    """
    Abstract base class for datasets that load, transform, and shuffle data before training.

    This class wraps a pandas DataFrame and provides additional functionality
    for data handling in machine learning contexts.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_data_description: DataDescription,
        target_data_description: DataDescription,
        id_column: str = "ID",
        use_augmentation: bool = False,
    ):
        """
        Initialize the Dataset.

        Args:
            data (pd.DataFrame): The input data as a pandas DataFrame.
            id_column (str, optional): The name of the ID column. Defaults to "ID".
        """
        self.data = data
        self.id_column = id_column
        self.input_data_description = input_data_description
        self.target_data_description = target_data_description
        self.use_augmentation = use_augmentation

    @property
    def validation_subset(self) -> "Dataset":
        if DatasetColumnNames.SUBSET.value in self.data.columns:
            return self[
                self[DatasetColumnNames.SUBSET.value] == Subset.VALIDATION.value
            ]
        raise MissingSubsetError()

    @property
    def test_subset(self) -> "Dataset":
        if DatasetColumnNames.SUBSET.value in self.data.columns:
            return self[self[DatasetColumnNames.SUBSET.value] == Subset.TEST.value]
        raise MissingSubsetError()

    @property
    def train_subset(self) -> "Dataset":
        if DatasetColumnNames.SUBSET.value in self.data.columns:
            train_subset = self[
                self[DatasetColumnNames.SUBSET.value] == Subset.TRAIN.value
            ]
            train_subset.use_augmentation = True
            return train_subset
        raise MissingSubsetError()

    def __getitem__(self, key):
        """
        Enable indexing and slicing of the dataset.

        This method allows using dataset[...] syntax, returning a new Dataset
        instance for DataFrame results, and the actual values for other types.

        Args:
            key: The index, column label, or condition to access data.

        Returns:
            Dataset or Any: A new Dataset instance if the result is a DataFrame,
                            otherwise the result itself.
        """
        result = self.data[key]
        return self.create_class_copy(result)

    def create_class_copy(self, result: any):
        if isinstance(result, pd.DataFrame):
            new_instance = self.__class__.__new__(self.__class__)
            new_instance.__dict__ = deepcopy(self.__dict__)
            new_instance.data = result
            return new_instance
        else:
            return result

    def __setitem__(self, key, value):
        """
        Enable item assignment for the dataset.

        This method allows using dataset[...] = ... syntax for assigning values,
        mirroring the behavior of the underlying DataFrame.

        Args:
            key: The index, column label, or condition to assign data to.
            value: The value to assign.
        """
        self.data[key] = value

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying DataFrame.

        This method allows using DataFrame methods and attributes directly
        on the Dataset instance. If the result is a DataFrame, it wraps it
        in a new Dataset instance.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute or method from the underlying DataFrame,
                 wrapped in a Dataset if it's a DataFrame.

        Raises:
            AttributeError: If the attribute is not found in the DataFrame.
        """
        attr = getattr(self.data, name)

        if name in ["iloc", "loc"]:
            return IndexerWrapper(self, attr)
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    new_instance = self.__class__.__new__(self.__class__)
                    new_instance.__dict__ = deepcopy(self.__dict__)
                    new_instance.data = result
                    return new_instance
                return result

            return wrapper
        return attr

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of rows in the underlying DataFrame.
        """
        return len(self.data)

    def __repr__(self) -> str:
        """
        String representation of the Dataset.

        Returns:
            str: A string describing the Dataset instance.
        """
        return f"{self.__class__.__name__}(rows={len(self)}, columns={list(self.data.columns)})"

    @abstractmethod
    def load_data(self, data: dict) -> dict:
        """Loads the data"""

    @abstractmethod
    def augment_data(self, data: dict) -> dict:
        """Augments the data"""

    @abstractmethod
    def transform_input_data(self, data: dict) -> any:
        """Transforms the data for the model"""

    @abstractmethod
    def transform_target_data(self, data: dict) -> any:
        """Transforms the target for the model"""

    def load_and_transform_data(self, index: int, include_target: bool = True) -> any:
        """Load and transform the data and returns it"""
        cur_data = self.get_datainfo(index)
        loaded_data = self.load_data(cur_data)
        if self.use_augmentation:
            loaded_data = self.augment_data(loaded_data)
        input_data = self.transform_input_data(loaded_data)
        if include_target:
            target_data = self.transform_target_data(loaded_data)
            return input_data, target_data
        return input_data

    def get_datainfo(self, index: int) -> dict:
        return self.data.iloc[index].to_dict()

    def get_datainfo_by_key(self, data_key):
        """Returns the data by the key (identifier of the data)"""
        return self.data[self.data[self.id_column] == data_key]

    def shuffle_and_sample(self) -> "Dataset":
        """Shuffles and samples the dataset"""
        return self.sample(frac=1).reset_index(drop=True)


class IndexerWrapper:
    """
    Wrapper class for DataFrame indexers like .iloc and .loc
    """

    def __init__(self, dataset, indexer):
        self.dataset = dataset
        self.indexer = indexer

    def __getitem__(self, key):
        result = self.indexer[key]
        if isinstance(result, pd.DataFrame):
            return self.dataset._create_new_instance(result)
        return result


class DatasetColumnNames(str, Enum):
    FILE_ID: str = "file_id"
    CLASS: str = "class"
    SUBSET: str = "subset"
    PREDICTION_RESULT: str = "prediction_result"

from abc import ABC, abstractmethod
from os.path import splitext

from hydra.utils import instantiate

import onnx
import pandas as pd
from onnx import ModelProto

from canaryml.datadescriptions import DataDescription
from canaryml.dataset import Dataset
from tqdm import tqdm

from canaryml.modelfactories.base_model_factory import ModelFactory
from canaryml.utils.io.location_config import LocationConfig, open_location
from canaryml.utils.io.read import read_yaml
from canaryml.utils.io.write import write_yaml


class WrongModelFileExtension(Exception):
    def __init__(
        self,
        message: str = "The model file extension of the provided path is not the expected one.",
        expected_extension: str | None = None,
    ):
        if expected_extension is not None:
            message += f" The expected model file extension is {expected_extension}."
        super().__init__(message)


class Trainer(ABC):
    def __init__(
        self,
        model_factory: ModelFactory,
        input_data_description: DataDescription,
        target_data_description: DataDescription,
    ):
        self.model_factory = model_factory
        self.model = model_factory.build_model(
            input_data_description, target_data_description
        )
        self.input_data_description = input_data_description
        self.target_data_description = target_data_description

    @abstractmethod
    def train(self, dataset: Dataset, epochs: int, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: Dataset, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def to_config(self) -> dict:
        raise NotImplementedError

    def serialize(self, location: LocationConfig):
        with open_location(location) as (file_system, filepath):
            config = self.to_config()
            config["_target_"] = self.__module__ + "." + self.__class__.__name__
            write_yaml(data=config, file_system=file_system, filepath=filepath)

    @classmethod
    def deserialize(cls, location: LocationConfig) -> "Trainer":
        with open_location(location) as (file_system, filepath):
            config = read_yaml(file_system=file_system, filepath=filepath)
            return instantiate(config, _convert_="all")

    def evaluate(
        self,
        dataset: Dataset,
        prediction_handlers: list[callable],
    ):
        predictions = self.predict(dataset=dataset.test_subset[:1000])
        output_data = []
        for cur_idx, prediction in enumerate(
            tqdm(predictions, desc="Evaluate predictions")
        ):
            cur_data = dataset.test_subset.get_datainfo(cur_idx)
            for prediction_handler in prediction_handlers:
                cur_data.update(
                    prediction_handler.handle_prediction(
                        prediction,
                        self.input_data_description,
                        self.target_data_description,
                    )
                )

            output_data.append(cur_data)

        return pd.DataFrame(output_data)

    @abstractmethod
    def save_as_onnx(self, model_path: str):
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass

    @abstractmethod
    def load(self, model_path: str) -> any:
        pass

    @staticmethod
    def load_as_onnx(model_path: str) -> ModelProto:
        if splitext(model_path)[1] != ".onnx":
            WrongModelFileExtension(expected_extension=".onnx")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        return model

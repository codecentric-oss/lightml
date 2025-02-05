from abc import ABC, abstractmethod

from canaryml.datadescriptions import DataDescription


class ModelFactory(ABC):
    @property
    def serializable_config(self) -> dict:
        config = self.to_config()
        config["_target_"] = self.__module__ + "." + self.__class__.__name__
        return config

    @abstractmethod
    def build_model(
        self,
        input_data_description: DataDescription,
        target_data_description: DataDescription,
    ) -> any:
        raise NotImplementedError

    @abstractmethod
    def to_config(self) -> dict:
        raise NotImplementedError

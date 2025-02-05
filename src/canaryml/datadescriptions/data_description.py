from abc import ABC, abstractmethod
from dataclasses import dataclass

from canaryml.utils.serialize import dataclass_to_dict


@dataclass
class DataDescription(ABC):
    @property
    def serializable_config(self) -> dict:
        return dataclass_to_dict(self)

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: "DataDescription") -> bool:
        raise NotImplementedError

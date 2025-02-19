from abc import ABC, abstractmethod

import numpy as np

from canaryml.datadescriptions import DataDescription


class PredictionHandler(ABC):
    @abstractmethod
    def handle_prediction(
        self,
        prediction: np.ndarray,
        input_data_description: DataDescription,
        target_data_description: DataDescription,
    ) -> dict:
        pass

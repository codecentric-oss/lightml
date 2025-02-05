import pandas as pd

from canaryml.dataset import Dataset
from lightml.trainer.trainer import Trainer


def train_and_evaluate(
    dataset: Dataset,
    trainer: Trainer,
    epochs: int,
    prediction_handlers: list[callable] | None = None,
) -> pd.DataFrame:
    trainer.train(
        dataset=dataset,
        epochs=epochs,
    )
    predictions = trainer.evaluate(
        dataset=dataset, prediction_handlers=prediction_handlers or []
    )
    return predictions

import inspect
from copy import deepcopy
from os.path import splitext
from typing import Any

import numpy as np
from keras import utils, Model, models, optimizers, losses, metrics as keras_metrics
from tqdm import tqdm
import tensorflow as tf

from canaryml.datadescriptions import DataDescription
from canaryml.dataset import Dataset
from canaryml.modelfactories.base_model_factory import ModelFactory
from canaryml.trainer.keras.exceptions import (
    NotAKerasModelError,
    UnknownOptimizerError,
    UnknownLossFunctionError,
    MissingMetricNameError,
    UnknownMetricError,
)
from canaryml.trainer.trainer import Trainer, WrongModelFileExtension


class KerasSequence(utils.Sequence):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        super().__init__()
        self.orig_dataset = dataset
        self.sampled_dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        The __len__ function is used to determine the number of batches in an epoch.
        Contrary to the __len__ function of the GenericDataset, this function
        returns the number of items per epoch.
        """
        batch_count, rest = divmod(len(self.sampled_dataset), self.batch_size)
        if rest > 0:
            batch_count += 1
        return batch_count

    def __getitem__(self, batch_index: int):
        """Returns the data of the batch at index"""
        start_idx = batch_index * self.batch_size
        end_idx = min(len(self.sampled_dataset), (batch_index + 1) * self.batch_size)
        net_inputs = []
        net_targets = []
        for cur_index in range(start_idx, end_idx):
            cur_input, cur_target = self.sampled_dataset.load_and_transform_data(
                cur_index
            )
            net_inputs.append(cur_input)
            net_targets.append(cur_target)

        net_inputs = np.array(net_inputs)
        net_targets = np.array(net_targets)
        return net_inputs, net_targets

    def on_epoch_end(self):
        """Shuffles the data if shuffle is True"""
        if self.shuffle:
            self.sampled_dataset = self.orig_dataset.shuffle_and_sample()


class KerasTrainer(Trainer):
    def __init__(
        self,
        model_factory: ModelFactory,
        input_data_description: DataDescription,
        target_data_description: DataDescription,
        batch_size: int,
        loss_fn: str | losses.Loss,
        optimizer: str | optimizers.Optimizer,
        metrics: list[dict[str, Any]] | list[callable] | None = None,
        loss_fn_kwargs: dict | None = None,
        optimizer_kwargs: dict | None = None,
        steps_per_epoch: int | None = None,
        validation_steps: int | None = None,
        shuffle: bool = True,
    ):
        super().__init__(
            model_factory=model_factory,
            input_data_description=input_data_description,
            target_data_description=target_data_description,
        )
        self.batch_size = batch_size
        self._get_loss_fn(loss_fn, loss_fn_kwargs)

        self._get_metrics(metrics)

        self._get_optimizer(optimizer, optimizer_kwargs)

        self.steps_per_epoch = steps_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.model: Model | None = None
        self.shuffle = shuffle

    def _get_optimizer(self, optimizer, optimizer_kwargs):  # TODO: Add test
        if isinstance(optimizer, str):
            self.optimizer_name = optimizer
            self.optimizer_kwargs = optimizer_kwargs
            self.optimizer = create_optimizer(
                optimizer_name=optimizer, optimizer_kwargs=optimizer_kwargs
            )
        else:
            self.optimizer_name = optimizer.name
            self.optimizer = optimizer
            self.optimizer_kwargs = self._get_kwargs_from_obj(obj=optimizer)

    def _get_metrics(self, metrics):  # TODO: Add test
        if isinstance(metrics, list) and len(metrics) > 0:
            if isinstance(metrics[0], dict):
                self.metric_names = [metric["metric_name"] for metric in metrics]
                self.metrics_list = metrics
                self.metrics = create_metrics(metrics=metrics)
            else:
                self.metrics = metrics
                self.metrics_list = []
                self.metric_names = []
                for metric in self.metrics:
                    self.metric_names.append(metric.__name__)
                    metric_kwargs = {"metric_name": metric.__name__}
                    metric_kwargs.update(**self._get_kwargs_from_obj(obj=metric))
                    self.metrics_list.append(metric_kwargs)
        else:
            self.metrics = []
            self.metrics_list = []

    def _get_loss_fn(self, loss_fn, loss_fn_kwargs):  # TODO: Add test
        if isinstance(loss_fn, str):
            self.loss_fn_name = loss_fn
            self.loss_fn = create_loss_function(
                loss_fn_name=self.loss_fn_name, loss_fn_kwargs=loss_fn_kwargs
            )
            self.loss_fn_kwargs = loss_fn_kwargs
        else:
            self.loss_fn = loss_fn
            self.loss_fn_name = loss_fn.name
            self.loss_fn_kwargs = self._get_kwargs_from_obj(obj=loss_fn)

    @staticmethod
    def _get_kwargs_from_obj(obj: any) -> dict:  # TODO: Add test
        kwargs = {}
        for kwarg in inspect.signature(
            obj.__class__ if not callable(obj) else obj
        ).parameters.keys():
            if kwarg not in ["args", "kwargs"]:
                try:
                    argument_value = getattr(obj, kwarg)
                except AttributeError:
                    continue
                kwargs[kwarg] = argument_value
        return kwargs

    def train(self, dataset: Dataset, epochs: int, **kwargs):
        from loguru import logger

        logger.add("Start to train model")

        self.build_model()
        return fit_model(
            model=self.model,
            dataset=dataset,
            epochs=epochs,
            batch_size=self.batch_size,
            loss_fn=self.loss_fn,
            metrics=self.metrics,
            optimizer=self.optimizer,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            **kwargs,
        )

    def build_model(self) -> Model:
        self.model = self.model_factory.build_model(
            input_data_description=self.input_data_description,
            target_data_description=self.target_data_description,
        )
        if not isinstance(self.model, Model):
            raise NotAKerasModelError()
        return self.model

    def predict(self, dataset: Dataset, **kwargs) -> np.ndarray:
        if not self.model:
            return np.ndarray([])

        return predict_dataset(dataset=dataset, model=self.model)

    def to_config(self) -> dict:
        config = {
            "model_factory": self.model_factory.serializable_config,
            "input_data_description": self.input_data_description.serializable_config,
            "target_data_description": self.target_data_description.serializable_config,
            "batch_size": self.batch_size,
            "loss_fn": self.loss_fn_name,
            "loss_fn_kwargs": self.loss_fn_kwargs,
            "metrics": self.metrics_list,
            "optimizer": self.optimizer_name,
            "optimizer_kwargs": self.optimizer_kwargs,
            "steps_per_epoch": self.steps_per_epoch,
            "validation_steps": self.validation_steps,
            "shuffle": self.shuffle,
        }
        return config

    def save_as_onnx(self, model_path: str):
        save_as_onnx(model=self.model, model_path=model_path)

    def save(self, model_path: str):
        if splitext(model_path)[1] != ".keras":
            raise WrongModelFileExtension(expected_extension=".keras")
        self.model.save(model_path)

    @classmethod
    def load(cls, model_path: str) -> Model:
        if splitext(model_path)[1] != ".keras":
            raise WrongModelFileExtension(expected_extension=".keras")
        return models.load_model(model_path)


def create_optimizer(
    optimizer_name: str, optimizer_kwargs: dict[str, any] | None = None
) -> optimizers.Optimizer:
    """
    Create a Keras optimizer based on the optimizer name and any additional keyword arguments.

    Returns:
        keras.optimizers.Optimizer: An instance of the specified optimizer.
    """
    if "_" in optimizer_name:
        optimizer_name = "".join(
            [
                optimizer_name_part.capitalize()
                for optimizer_name_part in optimizer_name.split("_")
            ]
        )
    if optimizer_name.islower():
        optimizer_name = optimizer_name.capitalize()
    # Get the optimizer constructor dynamically
    optimizer_class = getattr(optimizers, optimizer_name, None)

    if optimizer_class is not None:
        return optimizer_class(**optimizer_kwargs or {})
    else:
        raise UnknownOptimizerError()


def create_loss_function(
    loss_fn_name: str, loss_fn_kwargs: dict[str, any] | None = None
) -> losses.Loss:
    """
    Create a Keras loss function based on the loss name and any additional keyword arguments.

    Returns:
        keras.losses.Loss: An instance of the specified loss function.
    """
    if "_" in loss_fn_name:
        loss_fn_name = "".join(
            [
                loss_fn_name_part.capitalize()
                for loss_fn_name_part in loss_fn_name.split("_")
            ]
        )
    if loss_fn_name.islower():
        loss_fn_name = loss_fn_name.capitalize()
    # Get the loss function constructor dynamically
    loss_class = getattr(losses, loss_fn_name, None)

    if loss_class is not None:
        # Create an instance of the requested loss function
        return loss_class(**loss_fn_kwargs or {})
    else:
        raise UnknownLossFunctionError()


def create_metrics(metrics: list[dict[str, any]]) -> list[keras_metrics.Metric]:
    """
    Create a Keras metric based on the metric name and any additional keyword arguments.

    Returns:
        keras.metrics.Metric: An instance of the specified metric.
    """
    metrics_list: list[keras_metrics.Metric] = []
    for metric_dict in deepcopy(metrics):
        if "metric_name" not in metric_dict:
            raise MissingMetricNameError
        metric_name: str = metric_dict.pop("metric_name")
        if "_" in metric_name:
            metric_name = "".join(
                [
                    metric_name_part.capitalize()
                    for metric_name_part in metric_name.split("_")
                ]
            )
        if metric_name.islower():
            metric_name = metric_name.capitalize()

        # Get the metric class constructor dynamically
        metric_class = getattr(keras_metrics, metric_name, None)

        if metric_class is not None:
            # Create an instance of the requested metric
            metrics_list.append(metric_class(**metric_dict))
        else:
            raise UnknownMetricError()
    return metrics_list


def save_as_onnx(model: Model, model_path: str):
    import tf2onnx
    from loguru import logger

    logger.info("Saving model as ONNX model to %s", model_path)

    if splitext(model_path)[1] != ".onnx":
        raise WrongModelFileExtension(expected_extension=".onnx")

    tf2onnx.convert.from_keras(
        model=model,
        input_signature=[
            tf.TensorSpec(
                model_input.shape,
                dtype=model_input.dtype,
                name=model_input.name,
            )
            for model_input in model.inputs
        ],
        output_path=model_path,
    )


def fit_model(
    model: Model,
    dataset: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: losses.Loss,
    metrics,
    optimizer: optimizers.Optimizer,
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
    shuffle: bool = True,
    **kwargs,
) -> Model:
    train_sequence = KerasSequence(
        dataset=dataset.train_subset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    validation_sequence = KerasSequence(
        dataset=dataset.validation_subset, batch_size=batch_size
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics,
    )
    if steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, len(train_sequence))
    if validation_steps is not None:
        validation_steps = min(validation_steps, len(validation_sequence))
    model.fit(
        train_sequence,
        epochs=epochs,
        validation_data=validation_sequence,
        callbacks=[],
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        **kwargs,
    )
    return model


def predict_dataset(model: Model, dataset: Dataset) -> np.ndarray:
    predictions: np.ndarray | None = None

    for cur_idx in tqdm(range(len(dataset)), desc="Predict data"):
        input_data = dataset.load_and_transform_data(cur_idx, include_target=False)
        cur_prediction = model.predict_step(input_data[np.newaxis, :]).numpy()
        if predictions is None:
            predictions = cur_prediction
        else:
            predictions = np.concatenate((predictions, cur_prediction), axis=0)
    return predictions

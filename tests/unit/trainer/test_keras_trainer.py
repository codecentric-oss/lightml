from os.path import exists
from pathlib import Path

import pytest

from keras import optimizers, losses, Model, layers

from canaryml.datadescriptions import DataDescription
from canaryml.modelfactories.base_model_factory import ModelFactory
from canaryml.trainer.keras.exceptions import (
    UnknownOptimizerError,
    UnknownLossFunctionError,
    UnknownMetricError,
    MissingMetricNameError,
    NotAKerasModelError,
)
from canaryml.trainer.keras.keras_trainer import (
    create_optimizer,
    create_loss_function,
    create_metrics,
    save_as_onnx,
    KerasTrainer,
)
from canaryml.trainer.trainer import WrongModelFileExtension
from canaryml.utils.io.location_config import LocationConfig, open_location


@pytest.fixture
def mlp() -> Model:
    inputs = layers.Input(shape=(50,))

    x = layers.Dense(500, activation="relu")(inputs)
    x = layers.Dense(500, activation="relu")(x)
    x = layers.Dense(500, activation="relu")(x)

    outputs = layers.Dense(3)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


@pytest.fixture()
def model_serialization_location(tmp_path: Path) -> LocationConfig:
    return LocationConfig(uri=str(tmp_path / "serialized_model"))


@pytest.mark.parametrize(
    "optimizer_name,optimizer_kwargs",
    [("Adam", {}), ("adam", {}), ("Adam", {"learning_rate": 0.01}), ("SGD", {})],
)
def test_create_optimizer(optimizer_name: str, optimizer_kwargs: dict[str, any]):
    optimizer = create_optimizer(
        optimizer_name=optimizer_name, optimizer_kwargs=optimizer_kwargs
    )
    assert optimizer is not None
    assert isinstance(optimizer, optimizers.Optimizer)
    for keyword, argument in optimizer_kwargs.items():
        assert getattr(optimizer, keyword) == argument


def test_create_optimizer_unknown_optimizer():
    with pytest.raises(UnknownOptimizerError):
        create_optimizer(optimizer_name="unknown_optimizer")


@pytest.mark.parametrize(
    "loss_fn_name,loss_fn_kwargs",
    [
        ("BinaryCrossentropy", {}),
        ("BinaryCrossentropy", {"reduction": "sum"}),
        ("CategoricalFocalCrossentropy", {}),
        ("categorical_crossentropy", {}),
    ],
)
def test_create_loss_function(
    loss_fn_name: str, loss_fn_kwargs: dict[str, any]
) -> None:
    loss_fn = create_loss_function(
        loss_fn_name=loss_fn_name, loss_fn_kwargs=loss_fn_kwargs
    )
    assert loss_fn is not None
    assert isinstance(loss_fn, losses.Loss)
    for keyword, argument in loss_fn_kwargs.items():
        assert getattr(loss_fn, keyword) == argument


def test_create_loss_function_unknown_loss_fn():
    with pytest.raises(UnknownLossFunctionError):
        create_loss_function(loss_fn_name="unknown_loss_fn")


@pytest.mark.parametrize(
    "metrics",
    [
        ([]),
        ([{"metric_name": "Accuracy"}]),
        ([{"metric_name": "sparse_categorical_accuracy"}]),
        ([{"metric_name": "BinaryAccuracy", "threshold": 0.4}]),
        (
            [
                {"metric_name": "Accuracy"},
                {"metric_name": "BinaryAccuracy", "threshold": 0.6},
            ]
        ),
    ],
)
def test_create_metrics(metrics: list[dict[str, any]]):
    created_metrics = create_metrics(metrics=metrics)
    assert len(created_metrics) == len(metrics)
    for metric, created_metric in zip(metrics, created_metrics):
        metric.pop("metric_name")
        for keyword, argument in metric.items():
            assert getattr(created_metric, keyword) == argument


def test_create_metrics_unknown_metric():
    with pytest.raises(UnknownMetricError):
        create_metrics(metrics=[{"metric_name": "unknown_metric"}])


def test_create_metrics_missing_metric_name():
    with pytest.raises(MissingMetricNameError):
        create_metrics(metrics=[{"argument": 0.0}])


def test_save_as_onnx(mlp: Model, tmp_path: Path):
    filepath = tmp_path / "model.onnx"
    save_as_onnx(model=mlp, model_path=str(filepath))
    assert exists(filepath)


def test_save_as_onnx_wrong_file_extension(mlp: Model, tmp_path: Path):
    filepath = tmp_path / "model.onx"
    with pytest.raises(WrongModelFileExtension):
        save_as_onnx(model=mlp, model_path=str(filepath))


def test_keras_trainer_to_config(keras_trainer: KerasTrainer):
    config = keras_trainer.to_config()
    assert config["model_factory"] == keras_trainer.model_factory.serializable_config
    assert config["optimizer"] == keras_trainer.optimizer_name
    assert config["optimizer_kwargs"] == keras_trainer.optimizer_kwargs
    assert (
        config["input_data_description"]
        == keras_trainer.input_data_description.serializable_config
    )
    assert (
        config["target_data_description"]
        == keras_trainer.target_data_description.serializable_config
    )
    assert config["batch_size"] == keras_trainer.batch_size
    assert config["loss_fn"] == keras_trainer.loss_fn_name
    assert config["loss_fn_kwargs"] == keras_trainer.loss_fn_kwargs
    assert config["metrics"] == keras_trainer.metrics_list
    assert config["steps_per_epoch"] == keras_trainer.steps_per_epoch
    assert config["validation_steps"] == keras_trainer.validation_steps
    assert config["shuffle"] == keras_trainer.shuffle


def test_keras_trainer_serialize(
    keras_trainer: KerasTrainer,
    model_serialization_location: LocationConfig,
) -> None:
    keras_trainer.serialize(location=model_serialization_location)
    with open_location(model_serialization_location) as (
        serialized_model_file_system,
        serialized_model_filepath,
    ):
        assert serialized_model_file_system.exists(serialized_model_filepath)


def test_keras_trainer_deserialize(
    keras_trainer: KerasTrainer,
    model_serialization_location: LocationConfig,
) -> None:
    keras_trainer.serialize(location=model_serialization_location)
    loaded_trainer = keras_trainer.deserialize(location=model_serialization_location)
    assert isinstance(loaded_trainer, KerasTrainer)
    assert loaded_trainer.input_data_description == keras_trainer.input_data_description
    assert (
        loaded_trainer.target_data_description == keras_trainer.target_data_description
    )
    assert loaded_trainer.batch_size == keras_trainer.batch_size
    assert loaded_trainer.loss_fn.name == keras_trainer.loss_fn.name
    assert loaded_trainer.optimizer.name == keras_trainer.optimizer.name
    assert loaded_trainer.metrics == keras_trainer.metrics
    assert loaded_trainer.steps_per_epoch == keras_trainer.steps_per_epoch
    assert loaded_trainer.validation_steps == keras_trainer.validation_steps
    assert loaded_trainer.shuffle == keras_trainer.shuffle


def test_keras_trainer_build_model(keras_trainer: KerasTrainer):
    model = keras_trainer.build_model()
    assert isinstance(model, Model)


def test_keras_trainer_build_model_wrong_type(keras_trainer: KerasTrainer):
    class DummyModelFactory(ModelFactory):
        def build_model(
            self,
            input_data_description: DataDescription,
            target_data_description: DataDescription,
        ) -> any:
            return None

        def to_config(self) -> dict:
            return {}

    keras_trainer.model_factory = DummyModelFactory()
    with pytest.raises(NotAKerasModelError):
        keras_trainer.build_model()


def test_keras_trainer_save_model(keras_trainer: KerasTrainer, tmp_path: Path):
    keras_trainer.build_model()
    keras_trainer.save(model_path=str(tmp_path / "model.keras"))
    assert exists(tmp_path / "model.keras")


def test_keras_trainer_load_model(keras_trainer: KerasTrainer, tmp_path: Path):
    keras_trainer.build_model()
    keras_trainer.save(str(tmp_path / "model.keras"))
    model = KerasTrainer.load(model_path=str(tmp_path / "model.keras"))
    assert isinstance(model, Model)
    assert keras_trainer.model.name == model.name

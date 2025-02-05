import pytest

from canaryml.datadescriptions import ImageDataDescription
from canaryml.datadescriptions.category import CategoryDataDescription
from canaryml.modelfactories.base_model_factory import ModelFactory
from canaryml.trainer.keras.keras_trainer import KerasTrainer
from canaryml.utils.image.image_size import ImageSize

from keras import Model, layers


class TestModelFactory(ModelFactory):
    def build_model(
        self,
        input_data_description: ImageDataDescription,
        target_data_description: CategoryDataDescription,
    ) -> any:
        inputs = layers.Input(shape=input_data_description.get_tensor_shape())

        # First convolutional layer
        x = layers.Conv2D(32, kernel_size=(3, 3), padding="same")(inputs)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Second convolutional layer
        x = layers.Conv2D(64, kernel_size=(3, 3), padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Flatten the output from the convolutional layers
        x = layers.Flatten()(x)

        # Fully connected layer
        x = layers.Dense(128, activation="relu")(x)

        # Output layer with softmax activation for classification
        outputs = layers.Dense(len(target_data_description), activation="softmax")(x)

        # Creating the model
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def to_config(self) -> dict:
        return {}


@pytest.fixture
def category_names() -> list[str]:
    return ["a", "b", "c"]


@pytest.fixture
def image_data_description(category_names: list[str]) -> ImageDataDescription:
    return ImageDataDescription(
        image_size=ImageSize(height=256, width=256), channel_names=category_names
    )


@pytest.fixture
def category_data_description(category_names) -> CategoryDataDescription:
    return CategoryDataDescription(names=category_names)


@pytest.fixture
def model_factory() -> ModelFactory:
    return TestModelFactory()


@pytest.fixture
def keras_trainer(
    image_data_description: ImageDataDescription,
    category_data_description: CategoryDataDescription,
    model_factory: ModelFactory,
) -> KerasTrainer:
    return KerasTrainer(
        model_factory=model_factory,
        input_data_description=image_data_description,
        target_data_description=category_data_description,
        batch_size=32,
        loss_fn="categorical_crossentropy",
        optimizer="adam",
        metrics=[],
    )

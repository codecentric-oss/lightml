import dataclasses

import pytest

from canaryml.datadescriptions import ImageDataDescription
from canaryml.utils.image.image_size import ImageSize


def test_get_tensor_shape(image_data_description: ImageDataDescription):
    assert image_data_description.get_tensor_shape() == (
        *image_data_description.image_size.to_numpy_shape(),
        len(image_data_description.channel_names),
    )


def test_get_channel_count(image_data_description: ImageDataDescription):
    assert image_data_description.get_channel_count() == len(
        image_data_description.channel_names
    )


def test_repr(image_data_description: ImageDataDescription):
    repr_string = repr(image_data_description)
    assert ImageDataDescription.__name__ in repr_string
    for field in dataclasses.fields(image_data_description):
        assert field.name in repr_string


def test_len(image_data_description: ImageDataDescription):
    assert len(image_data_description) == image_data_description.get_channel_count()


def test_iter(image_data_description: ImageDataDescription):
    idx_count = 0
    for idx, channel_name in enumerate(image_data_description):
        idx_count += 1
        assert channel_name in image_data_description.channel_names
    assert idx_count == len(image_data_description.channel_names)


def test_str(image_data_description: ImageDataDescription):
    str_string = str(image_data_description)
    for field in dataclasses.fields(image_data_description):
        field_name_in_str = " ".join(
            [field_name_part.capitalize() for field_name_part in field.name.split("_")]
        )
        assert field_name_in_str in str_string


def test_eq():
    first_data_description = ImageDataDescription(
        image_size=ImageSize(height=256, width=256), channel_names=["a", "b", "c"]
    )
    second_data_description = ImageDataDescription(
        image_size=ImageSize(height=256, width=256), channel_names=["a", "b", "c"]
    )
    assert first_data_description == second_data_description


@pytest.mark.parametrize(
    "first_data_description,second_data_description",
    [
        (
            ImageDataDescription(
                channel_names=["a", "b", "c"],
                image_size=ImageSize(height=256, width=256),
            ),
            ImageDataDescription(
                channel_names=["a", "b"], image_size=ImageSize(height=256, width=256)
            ),
        ),
        (
            ImageDataDescription(
                channel_names=["a", "b", "c"],
                image_size=ImageSize(height=256, width=512),
            ),
            ImageDataDescription(
                channel_names=["a", "b", "c"],
                image_size=ImageSize(height=256, width=256),
            ),
        ),
    ],
)
def test_not_eq(
    first_data_description: ImageDataDescription,
    second_data_description: ImageDataDescription,
):
    assert first_data_description != second_data_description

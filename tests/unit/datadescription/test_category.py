import dataclasses

import pytest

from canaryml.datadescriptions.category import CategoryDataDescription


def test_get_category_index(
    category_data_description: CategoryDataDescription, category_names: list[str]
):
    for index, category_name in enumerate(category_names):
        assert category_data_description.get_category_index(name=category_name) == index


def test_repr(category_data_description: CategoryDataDescription):
    repr_string = repr(category_data_description)
    assert CategoryDataDescription.__name__ in repr_string
    for field in dataclasses.fields(category_data_description):
        assert field.name in repr_string


def test_len(category_data_description: CategoryDataDescription):
    assert len(category_data_description) == len(category_data_description.names)


def test_iter(category_data_description: CategoryDataDescription):
    idx_count = 0
    for idx, name in enumerate(category_data_description):
        idx_count += 1
        assert name in category_data_description.names
    assert idx_count == len(category_data_description.names)


def test_str(category_data_description: CategoryDataDescription):
    str_string = str(category_data_description)
    for field in dataclasses.fields(category_data_description):
        field_name_in_str = " ".join(
            [field_name_part.capitalize() for field_name_part in field.name.split("_")]
        )
        assert field_name_in_str in str_string


def test_eq():
    first_data_description = CategoryDataDescription(names=["a", "b", "c"])
    second_data_description = CategoryDataDescription(names=["a", "b", "c"])
    assert first_data_description == second_data_description


@pytest.mark.parametrize(
    "first_data_description,second_data_description",
    [
        (
            CategoryDataDescription(
                names=["a", "b", "c"],
            ),
            CategoryDataDescription(names=["a", "b"]),
        ),
    ],
)
def test_not_eq(
    first_data_description: CategoryDataDescription,
    second_data_description: CategoryDataDescription,
):
    assert first_data_description != second_data_description

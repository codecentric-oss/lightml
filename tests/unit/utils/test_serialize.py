import pytest
from dataclasses import dataclass

from canaryml.utils.serialize import dataclass_to_dict


# Sample dataclasses for testing
@dataclass
class Address:
    street: str
    city: str


@dataclass
class User:
    name: str
    age: int
    address: Address


@pytest.fixture
def user():
    return User(
        name="Alice", age=30, address=Address(street="123 Main St", city="Wonderland")
    )


def test_dataclass_to_dict_basic(user):
    result = dataclass_to_dict(user)
    expected = {
        "name": "Alice",
        "age": 30,
        "address": {
            "_target_": Address.__module__ + "." + Address.__name__,
            "street": "123 Main St",
            "city": "Wonderland",
        },
        "_target_": User.__module__ + "." + User.__name__,
    }
    assert result == expected


def test_dataclass_to_dict_no_target(user):
    result = dataclass_to_dict(user, add_target=False)
    expected = {
        "name": "Alice",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "Wonderland",
        },
    }
    assert result == expected


def test_dataclass_to_dict_list():
    users = [
        User(
            name="Alice",
            age=30,
            address=Address(street="123 Main St", city="Wonderland"),
        ),
        User(
            name="Bob", age=25, address=Address(street="456 Elm St", city="Builderland")
        ),
    ]

    result = dataclass_to_dict(users)
    expected = [
        {
            "name": "Alice",
            "age": 30,
            "address": {
                "_target_": Address.__module__ + "." + Address.__name__,
                "street": "123 Main St",
                "city": "Wonderland",
            },
            "_target_": User.__module__ + "." + User.__name__,
        },
        {
            "name": "Bob",
            "age": 25,
            "address": {
                "_target_": Address.__module__ + "." + Address.__name__,
                "street": "456 Elm St",
                "city": "Builderland",
            },
            "_target_": User.__module__ + "." + User.__name__,
        },
    ]
    assert result == expected


def test_dataclass_to_dict_dict():
    user_dict = {
        "alice": User(
            name="Alice",
            age=30,
            address=Address(street="123 Main St", city="Wonderland"),
        ),
        "bob": User(
            name="Bob", age=25, address=Address(street="456 Elm St", city="Builderland")
        ),
    }

    result = dataclass_to_dict(user_dict)
    expected = {
        "alice": {
            "name": "Alice",
            "age": 30,
            "address": {
                "_target_": Address.__module__ + "." + Address.__name__,
                "street": "123 Main St",
                "city": "Wonderland",
            },
            "_target_": User.__module__ + "." + User.__name__,
        },
        "bob": {
            "name": "Bob",
            "age": 25,
            "address": {
                "_target_": Address.__module__ + "." + Address.__name__,
                "street": "456 Elm St",
                "city": "Builderland",
            },
            "_target_": User.__module__ + "." + User.__name__,
        },
    }
    assert result == expected

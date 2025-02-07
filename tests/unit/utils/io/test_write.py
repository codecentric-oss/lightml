import os
from os.path import join
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import pytest

from canaryml.utils.io.location_config import join_fs_path, open_location
from canaryml.utils.io.write import (
    write_parquet,
    write_image,
    write_json,
    write_csv,
    write_yaml,
)


@pytest.fixture
def location_config(tmp_path: Path) -> dict:
    return {"uri": join(str(tmp_path), "test_io_write")}


@pytest.fixture
def data_to_be_written() -> pd.DataFrame:
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})


def test_write_parquet(location_config: dict, data_to_be_written: pd.DataFrame):
    with open_location(location_config) as (file_system, root_directory):
        file_path = join_fs_path(file_system, root_directory, "data.parquet")
        write_parquet(
            data=data_to_be_written,
            filepath=file_path,
            file_system=file_system,
        )
    assert os.path.exists(file_path)


def test_write_image(location_config: dict):
    with open_location(location_config) as (file_system, root_directory):
        file_path = join_fs_path(file_system, root_directory, "image.png")
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        write_image(image=image, filepath=file_path, file_system=file_system)

    assert os.path.exists(file_path)


def test_write_json(location_config: dict, data_to_be_written: pd.DataFrame):
    with open_location(location_config) as (file_system, root_directory):
        file_path = join_fs_path(file_system, root_directory, "data.json")
        write_json(
            data=data_to_be_written.to_dict(),
            filepath=file_path,
            file_system=file_system,
        )
    assert os.path.exists(file_path)


def test_write_csv(location_config: dict, data_to_be_written: pd.DataFrame):
    with open_location(location_config) as (file_system, root_directory):
        file_path = join_fs_path(file_system, root_directory, "data.csv")
        write_csv(
            data=data_to_be_written,
            filepath=file_path,
            file_system=file_system,
        )
    assert os.path.exists(file_path)


def test_write_yaml(location_config: dict, data_to_be_written: pd.DataFrame):
    with open_location(location_config) as (file_system, root_directory):
        file_path = join_fs_path(file_system, root_directory, "data.yaml")
        write_yaml(
            data=data_to_be_written.to_dict(),
            filepath=file_path,
            file_system=file_system,
        )
    assert os.path.exists(file_path)

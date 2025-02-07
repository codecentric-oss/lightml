from os.path import join
from pathlib import Path

import fastparquet
from PIL import Image
import numpy as np
import pandas as pd
import pytest
import yaml

from canaryml.utils.image.image_size import ImageSize
from canaryml.utils.io.location_config import open_location, join_fs_path
from canaryml.utils.io.read import (
    list_dir,
    read_parquet,
    read_csv,
    read_yaml,
    read_image,
    find_and_read_file,
    read_json,
)


@pytest.fixture()
def location_config(tmp_path: Path) -> dict:
    return {"uri": join(str(tmp_path), "test_io_read")}


@pytest.fixture(autouse=True)
def write_data(location_config: dict):
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    with open_location(location_config) as (file_system, root_directory):
        filepath = join_fs_path(file_system, root_directory)
        img = Image.fromarray(image)
        img.save(f"{filepath}/image.png")
        data.to_csv(f"{filepath}/data.csv")
        data.to_json(f"{filepath}/data.json")
        fastparquet.write(data=data, filename=f"{filepath}/data.parquet")
        with open(f"{filepath}/data.yaml", "w") as file:
            yaml.dump(data.to_dict(), file)


def test_list_dir(location_config: dict):
    expected_files = ["image.png", "data.csv", "data.json", "data.parquet", "data.yaml"]
    with open_location(location_config) as (file_system, root_directory):
        found_files = list_dir(path=root_directory, file_system=file_system)
        assert sorted(found_files) == sorted(expected_files)


@pytest.mark.parametrize(
    "file_extensions,expected_files",
    [
        ([".png"], ["image.png"]),
        ([".png", ".csv"], ["image.png", "data.csv"]),
        ([".csv", ".json"], ["data.csv", "data.json"]),
        ([], ["image.png", "data.csv", "data.json", "data.parquet", "data.yaml"]),
        (None, ["image.png", "data.csv", "data.json", "data.parquet", "data.yaml"]),
    ],
)
def test_list_dir_file_extensions(
    location_config: dict, file_extensions: list[str], expected_files: list[str]
):
    with open_location(location_config) as (file_system, root_directory):
        found_files = list_dir(
            path=root_directory, file_system=file_system, filter_ext=file_extensions
        )
        assert sorted(found_files) == sorted(expected_files)


def test_read_parquet(location_config: dict):
    with open_location(location_config) as (file_system, root_directory):
        data = read_parquet(
            filepath=join_fs_path(file_system, root_directory, "data.parquet"),
            file_system=file_system,
        )
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert data.columns.tolist() == ["col1", "col2"]


def test_read_yaml(location_config: dict):
    with open_location(location_config) as (file_system, root_directory):
        data = read_yaml(
            filepath=join_fs_path(file_system, root_directory, "data.yaml"),
            file_system=file_system,
        )
    assert isinstance(data, dict)
    data = pd.DataFrame(data=data)
    assert len(data) == 3
    assert data.columns.tolist() == ["col1", "col2"]


def test_read_json(location_config: dict):
    with open_location(location_config) as (file_system, root_directory):
        data = read_json(
            filepath=join_fs_path(file_system, root_directory, "data.json"),
            file_system=file_system,
        )
    assert isinstance(data, dict)
    data = pd.DataFrame(data=data)
    assert len(data) == 3
    assert data.columns.tolist() == ["col1", "col2"]


@pytest.mark.parametrize(
    "expected_columns,function_arguments",
    [([["Unnamed: 0", "col1", "col2"], {}]), ([["col1", "col2"], {"index_col": 0}])],
)
def test_read_csv(
    location_config: dict, expected_columns: list[str], function_arguments: dict
):
    with open_location(location_config) as (file_system, root_directory):
        data = read_csv(
            filepath=join_fs_path(file_system, root_directory, "data.csv"),
            file_system=file_system,
            **function_arguments,
        )
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert data.columns.tolist() == expected_columns


def test_read_image(location_config: dict):
    with open_location(location_config) as (file_system, root_directory):
        image = read_image(
            filepath=join_fs_path(file_system, root_directory, "image.png"),
            file_system=file_system,
        )
    assert isinstance(image, Image.Image)
    assert image.size == (100, 100)


def test_read_image_as_array(location_config: dict):
    with open_location(location_config) as (file_system, root_directory):
        image = read_image(
            filepath=join_fs_path(file_system, root_directory, "image.png"),
            file_system=file_system,
            as_array=True,
        )
    assert isinstance(image, np.ndarray)
    assert image.shape == (100, 100, 3)
    assert image.dtype == np.uint8


@pytest.mark.parametrize("as_array", [True, False])
def test_read_image_resize(location_config: dict, as_array: bool):
    with open_location(location_config) as (file_system, root_directory):
        image = read_image(
            filepath=join_fs_path(file_system, root_directory, "image.png"),
            file_system=file_system,
            resize_to=ImageSize(50, 50),
            as_array=as_array,
        )
    if as_array:
        assert isinstance(image, np.ndarray)
        assert image.shape == (50, 50, 3)
        assert image.dtype == np.uint8
    else:
        assert isinstance(image, Image.Image)
        assert image.size == (50, 50)


@pytest.mark.parametrize(
    "read_function,file_name,expected_data_type",
    [
        (read_image, "image.png", Image.Image),
        (read_yaml, "data.yaml", dict),
        (read_parquet, "data.parquet", pd.DataFrame),
        (read_csv, "data.csv", pd.DataFrame),
        (read_json, "data.json", dict),
    ],
)
def test_find_and_read_file(
    location_config: dict,
    read_function: callable,
    file_name: str,
    expected_data_type: type,
):
    with open_location(location_config) as (file_system, root_directory):
        filepath = join_fs_path(file_system, root_directory, file_name)
        returned_filepath, file = find_and_read_file(
            filepath=filepath,
            file_system=file_system,
            read_func=read_function,
        )
        assert isinstance(file, expected_data_type)
        assert returned_filepath == filepath


def test_find_and_read_file_search_paths(
    location_config: dict,
):
    with open_location(location_config) as (file_system, root_directory):
        returned_filepath, file = find_and_read_file(
            filepath="data.parquet",
            search_paths=[root_directory],
            file_system=file_system,
            read_func=read_parquet,
        )
        assert isinstance(file, pd.DataFrame)
        assert returned_filepath == join_fs_path(
            file_system, root_directory, "data.parquet"
        )


def test_find_and_read_file_file_not_found(location_config: dict):
    with open_location(location_config) as (file_system, _):
        with pytest.raises(FileNotFoundError):
            find_and_read_file(
                filepath=join_fs_path(file_system, "unknown_directory", "data.parquet"),
                file_system=file_system,
                read_func=read_parquet,
            )

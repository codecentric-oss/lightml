from os.path import join

import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from canaryml.utils.io.location_config import (
    LocationConfig,
    join_location_w_path,
    get_location_uri,
    join_fs_path,
)


def test_join_location_w_path():
    s3_path = "s3://bucket"
    filepath = "path/to/file"
    old_config = LocationConfig(uri=s3_path, fs_args={"an_arg": "value"})
    new_config = join_location_w_path(old_config, filepath)
    new_config_multiple = join_location_w_path(old_config, ["foldername", filepath])
    assert new_config.uri == join(s3_path, filepath)
    assert new_config.fs_args == old_config.fs_args
    assert new_config_multiple.uri == join(s3_path, "foldername", filepath)
    assert new_config_multiple.fs_args == old_config.fs_args
    assert old_config.uri == s3_path


@pytest.mark.parametrize(
    "file_system, paths, expected",
    [
        (
            LocalFileSystem(),
            ["folder1", "folder2", "file.txt"],
            "folder1/folder2/file.txt",
        ),
        (LocalFileSystem(), ["folder1", "", "file.txt"], "folder1/file.txt"),
        (LocalFileSystem(), ["", "", ""], ""),
        (LocalFileSystem(), ["only_one_path"], "only_one_path"),
        (
            MemoryFileSystem(),
            ["folder1", "folder2", "file.txt"],
            "folder1/folder2/file.txt",
        ),
    ],
)
def test_join_fs_path(file_system, paths, expected):
    result = join_fs_path(file_system, *paths)
    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_get_location_uri_with_fs_path_config():
    config = LocationConfig(uri="s3://my-bucket")
    assert get_location_uri(config) == "s3://my-bucket"


def test_get_location_uri_with_dict():
    config_dict = {"uri": "gs://my-bucket"}
    assert get_location_uri(config_dict) == "gs://my-bucket"


def test_get_location_uri_with_invalid_input():
    with pytest.raises(TypeError):
        get_location_uri({"invalid input": "invalid"})


def test_get_location_uri_with_missing_uri():
    with pytest.raises(TypeError):
        config_dict = {"fs_args": {"key": "value"}}
        config = LocationConfig(**config_dict)
        get_location_uri(config)

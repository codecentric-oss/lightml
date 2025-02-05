import json
from os.path import dirname

import fastparquet
import yaml
from PIL import Image
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
import pandas as pd


def write_parquet(
    dataframe: pd.DataFrame,
    filepath: str,
    compression: str | None = "gzip",
    file_system: AbstractFileSystem | None = None,
    **kwargs,
):
    """
    Writes dataframe to parquet file with optional AbstractFileSystem given

    Args:
        dataframe: Dataframe to write to parquet file
        filepath: Path to save the parquet file to
        compression: Compression method
        file_system: Allow the function to be used with different file systems; default = local
        **kwargs: additional arguments for fastparquet.write function
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    cur_fs.mkdirs(
        dirname(filepath),
        exist_ok=True,
    )
    fastparquet.write(
        filepath,
        dataframe,
        open_with=cur_fs.open,
        compression=compression,
        **kwargs,
    )


def write_image(
    image: Image.Image,
    filepath: str,
    file_system: AbstractFileSystem | None = None,
    **kwargs,
):
    """
    Saves image to filepath with optional AbstractFileSystem given

    Args:
        image: Image object
        filepath: Path to save the image to
        file_system: Allow the function to be used with different file systems; default = local
        **kwargs: additional arguments for Image.save function
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    cur_fs.mkdirs(
        dirname(filepath),
        exist_ok=True,
    )
    with file_system.open(filepath, "wb") as file:
        file_format = filepath.rsplit(".")[-1]
        image.save(file, format=file_format, **kwargs)


def write_json(
    data: dict,
    filepath: str,
    file_system: AbstractFileSystem | None = None,
    **kwargs,
):
    """
    Writes dictionary to json with optional AbstractFileSystem given

    Args:
        data: dictionary to be saved as json
        filepath: path to save the json file to
        file_system: Allow the function to be used with different file systems; default = local
        **kwargs: additional arguments for json.dump function
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    cur_fs.mkdirs(
        dirname(filepath),
        exist_ok=True,
    )
    with cur_fs.open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, **kwargs)


def write_csv(
    data: pd.DataFrame,
    filepath: str,
    file_system: AbstractFileSystem | None = None,
    **kwargs,
):
    """
    Writes dataframe to csv file with optional AbstractFileSystem given

    Args:
        data: Dataframe to write to csv file
        filepath: Path to save the csv file to
        file_system: Allow the function to be used with different file systems; default = local
        **kwargs: additional arguments for data.to_csv function
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    cur_fs.mkdirs(
        dirname(filepath),
        exist_ok=True,
    )
    with cur_fs.open(filepath, "w", encoding="utf-8") as file:
        data.to_csv(file, index=False, **kwargs)


def write_yaml(
    data: dict,
    filepath: str,
    file_system: AbstractFileSystem | None = None,
    **kwargs,
):
    """
    Writes dictionary to yaml with optional AbstractFileSystem given

    Args:
        data: dictionary to be saved as yaml
        filepath: path to save the yaml file to
        file_system: Allow the function to be used with different file systems; default = local
        **kwargs: additional arguments for yaml.dump function
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    cur_fs.mkdirs(
        dirname(filepath),
        exist_ok=True,
    )
    with cur_fs.open(filepath, "w", encoding="utf-8") as file:
        yaml.dump(data, file, Dumper=yaml.SafeDumper, **kwargs)

# Copyright (c) codecentric AG 2025. All Rights Reserved.
#
# Licensed under the MIT License. See the LICENSE file in the project root for more information.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# 1. The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
# 2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.


"""Module for helper functions for io operations"""

import json
from os.path import join, relpath, splitext
from typing import List, Optional, Any, Tuple

import fastparquet
import numpy as np
import pandas as pd
import yaml
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from PIL import Image

from lightml.utils.images.image_size import ImageSize
from lightml.utils.io.location import join_fs_path


def list_dir(
    path: str,
    return_full_path: bool = False,
    recursive: bool = False,
    file_system: Optional[AbstractFileSystem] = None,
    filter_ext: Optional[List[str]] = None,
) -> List[str]:
    """
    Returns a list of files in a directory

    Args:
        path: path to directory, which should be listed
        return_full_path: Returns full filepaths (True) or relative path (False)
        recursive: Determine if the function should look into subfolders
        file_system: Allow the function to be used with different file systems; default = local
        filter_ext: List of file extension to filter for; default = all files

    Returns:
        A list of files in the specified directory
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    files: List[str] = [
        relpath(cur_file, path) for cur_file in list(cur_fs.listdir(path, detail=False))
    ]
    if recursive:
        folders = [
            cur_folder for cur_folder in files if cur_fs.isdir(join(path, cur_folder))
        ]
        for cur_folder in folders:
            files += [
                join(cur_folder, cur_file)
                for cur_file in list_dir(
                    join(path, cur_folder), False, True, file_system=cur_fs
                )
            ]

    if filter_ext is not None:
        files = [cur_file for cur_file in files if splitext(cur_file)[1] in filter_ext]

    if return_full_path:
        files = [join(path, cur_file) for cur_file in files]

    return files


def read_parquet(
    filepath: str, file_system: Optional[AbstractFileSystem] = None
) -> pd.DataFrame:
    """
    Reads parquet with optional AbstractFileSystem given

    Args:
        filepath: path to parquet file
        file_system: Allow the function to be used with different file systems; default = local

    Returns:
        dataframe from parquet file
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    if not cur_fs.exists(filepath):
        raise FileNotFoundError(f"Parquetfile not found: {filepath}")
    return fastparquet.ParquetFile(filepath, fs=cur_fs).to_pandas()


def read_yaml(filepath: str, file_system: Optional[AbstractFileSystem] = None) -> dict:
    """
    Reads a yaml file with optional AbstractFileSystem given

    Args:
        filepath: path to yaml file
        file_system: Allow the function to be used with different file systems; default = local

    Returns:
        Content of yaml as dictionary
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    if not cur_fs.exists(filepath):
        raise FileNotFoundError(f"Yamlfile not found: {filepath}")
    with cur_fs.open(filepath, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def read_json(filepath: str, file_system: Optional[AbstractFileSystem] = None) -> dict:
    """
    Reads a json file with optional AbstractFileSystem given

    Args:
        filepath: path to json file
        file_system: Allow the function to be used with different file systems; default = local

    Returns:
        Content of json as dictionary
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    if not cur_fs.exists(filepath):
        raise FileNotFoundError(f"Yamlfile not found: {filepath}")
    with cur_fs.open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(
    filepath: str, file_system: Optional[AbstractFileSystem] = None, **kwargs
) -> pd.DataFrame:
    """Reads csv with optional AbstractFileSystem given

    Args:
        filepath: path to csv file
        file_system: Allow the function to be used with different file systems; default = local
        ***kwargs: additional arguments for pd.read_csv function

    Returns:
        dataframe from csv file"""
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    if not cur_fs.exists(filepath):
        raise FileNotFoundError(f"CSV not found: {filepath}")
    with cur_fs.open(filepath, "r", encoding="utf-8") as file:
        return pd.read_csv(file, **kwargs)


def read_image(
    filepath: str,
    file_system: Optional[AbstractFileSystem] = None,
    as_array: bool = False,
    resize_to: ImageSize | None = None,
    **kwargs,
) -> Image.Image | np.ndarray:
    """Reads image with optional AbstractFileSystem given

    Args:
        filepath: Path to load the image from
        file_system: Allow the function to be used with different file systems; default = local
        **kwargs: additional arguments for Image.open function

    Returns:
        loaded image object
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    if not cur_fs.exists(filepath):
        raise FileNotFoundError(f"ImageFile not found: {filepath}")

    with file_system.open(filepath, "rb") as file:
        image = Image.open(file, **kwargs).copy()
        if resize_to is not None:
            image = image.resize(size=resize_to.to_pil_size())
        if as_array:
            image = np.asarray(image)

            if image.dtype == bool:
                np_array = image.astype(np.uint8)
                np_array *= 255

    return image


def find_and_read_file(
    filepath: str,
    read_func,
    search_paths: Optional[List[str]] = None,
    file_system: Optional[AbstractFileSystem] = None,
    **kwargs,
) -> Tuple[str, Any]:
    """
    Tries to find a file in a list of search paths and reads it with given read function

    Args:
        filepath: path to file
        search_paths: list of paths to search for file
        read_func: function to read the file
        file_system: Allow the function to be used with different file systems; default = local

    Returns:
        Content of file
    """
    cur_fs: AbstractFileSystem = file_system or LocalFileSystem()
    search_paths = search_paths or []
    if cur_fs.exists(filepath):
        return filepath, read_func(filepath, file_system=cur_fs, **kwargs)
    for path in search_paths:
        cur_path = join_fs_path(cur_fs, path, filepath)
        if cur_fs.exists(cur_path):
            return cur_path, read_func(cur_path, file_system=cur_fs, **kwargs)
    raise FileNotFoundError(f"File not found: {filepath}")

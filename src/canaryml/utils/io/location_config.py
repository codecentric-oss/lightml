from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from os.path import join
from typing import Iterator
from fsspec.core import url_to_fs

from fsspec import AbstractFileSystem

FSPath = tuple[AbstractFileSystem, str]


@dataclass
class LocationConfig:  # pylint: disable=too-few-public-methods
    """Access description for a remote storage location. The description is targeted at fsspec_.

      Attributes:
          uri: RL to remote storage as expected by fsspec_.
          fs_args: Optional filesystem arguments to be passed to fsspec_, see e.g.
    https://filesystem-spec.readthedocs.io/en/latest/api.html#built-in-implementations
          credentials: Optional credentials to be passed as filesystem arguments to fsspec_.
    """

    uri: str
    fs_args: dict[str, any] = field(default_factory=dict)
    credentials: dict[str, any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Returns string representation without credential values"""
        info = asdict(self)
        info["credentials"] = list(info["credentials"])
        return str(info)


def join_location_w_path(
    location: LocationConfig | dict, path: list[str] | str
) -> LocationConfig:
    """Returns joined LocationConfig with one or more path objects"""
    parsed_config = parse_location(location)
    copied_config = deepcopy(parsed_config)
    path_list = path if isinstance(path, list) else [path]
    copied_config.uri = join(copied_config.uri, *path_list)
    return copied_config


def join_fs_path(file_system: AbstractFileSystem, *paths: str) -> str:
    """Returns joined given paths with the fsspec specific path seperator"""
    paths = [path for path in paths if len(path) > 0]
    return file_system.sep.join(paths)


def get_location_uri(location: LocationConfig | dict) -> str:
    """Returns the URI of a LocationConfig."""
    parsed_config = parse_location(location)
    return parsed_config.uri


def parse_location(location: LocationConfig | dict) -> LocationConfig:
    """Returns the LocationConfig object"""
    parsed_config = (
        location if isinstance(location, LocationConfig) else LocationConfig(**location)
    )
    return parsed_config


@contextmanager
def open_location(config: LocationConfig | dict[str, any]) -> Iterator[FSPath]:
    """
    Creates a filesystem and path from configuration as a single context manager.
    The filesystem is "closed" (i.e. open connections are closed) when the context is left.
    """
    parsed_config = parse_location(location=config)
    credentials = deepcopy(parsed_config.credentials)
    fs_args = deepcopy(parsed_config.fs_args)
    filesystem, path = url_to_fs(parsed_config.uri, **credentials, **fs_args)
    try:
        yield filesystem, path
    finally:
        del filesystem

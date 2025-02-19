from dataclasses import is_dataclass, fields
from typing import Any

from hydra.utils import instantiate


def dataclass_to_dict(
    obj: dict[str, Any] | list | Any, add_target: bool = True
) -> dict[str, any] | list:
    """
    Converts a dataclass instance to a dictionary.
    """
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            result[field.name] = dataclass_to_dict(getattr(obj, field.name), add_target)
        if add_target:
            result["_target_"] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
        return result
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v, add_target) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(v, add_target) for v in obj]
    else:
        return obj


def dict_to_dataclass(config_dict: dict[str, any]):
    """Converts a dictionary to a dataclass instance."""
    return instantiate(config_dict, _convert_="all")

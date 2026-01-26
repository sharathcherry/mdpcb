"""Utility helpers for loading prediction models."""

from __future__ import annotations

import pickle
from typing import Dict, Tuple

ModelMap = Dict[str, str]
LoadedModels = Dict[str, object]
FailedModels = Dict[str, str]


def load_models(model_files: ModelMap) -> Tuple[LoadedModels, FailedModels]:
    """Load serialized models defined in ``model_files``.

    Parameters
    ----------
    model_files:
        Mapping of attribute names to file paths.

    Returns
    -------
    tuple of dicts
        ``(loaded_models, failed_models)`` where *loaded_models* maps attribute
        names to deserialized objects and *failed_models* maps attribute names to
        the failure reason.
    """

    loaded_models: LoadedModels = {}
    failed_models: FailedModels = {}

    for attr_name, model_path in model_files.items():
        try:
            with open(model_path, "rb") as fh:
                loaded_models[attr_name] = pickle.load(fh)
        except FileNotFoundError:
            failed_models[attr_name] = "file not found"
        except Exception as exc:  # noqa: BLE001 - expose precise failure
            failed_models[attr_name] = str(exc)

    return loaded_models, failed_models

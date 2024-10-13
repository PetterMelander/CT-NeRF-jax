import json
from pathlib import Path

import numpy as np


def get_data_dir() -> Path:
    return Path(__file__).parents[1] / "data"


def get_model_dir() -> Path:
    return Path(__file__).parents[1] / "models"


def convert_arrays_to_lists(d: dict) -> dict:
    """
    Utility function for converting numpy arrays to lists in a nestled dict. Used for json
    dumping metadata.
    """
    if isinstance(d, dict):
        return {k: convert_arrays_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_arrays_to_lists(i) for i in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d


def get_dataset_metadata(dataset_path: Path) -> dict:
    with open(dataset_path / "meta.json", "r") as f:
        metadata = json.load(f)
    return metadata

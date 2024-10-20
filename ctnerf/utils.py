"""Utility functions."""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def get_data_dir() -> Path:
    """Get the path to the data directory."""
    return Path(__file__).parents[1] / "data"


def get_model_dir() -> Path:
    """Get the path to the models directory."""
    return Path(__file__).parents[1] / "models"


def convert_arrays_to_lists(d: dict) -> dict:
    """Convert numpy arrays to lists in a nestled dict.

    Utility function for converting numpy arrays to lists in a nestled dict. Used for json dumping
    metadata.
    """
    if isinstance(d, dict):
        return {k: convert_arrays_to_lists(v) for k, v in d.items()}
    if isinstance(d, list):
        return [convert_arrays_to_lists(i) for i in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def get_dataset_metadata(dataset_path: Path) -> dict:
    """Read the metadata for a given dataset from the 'meta.json' file in the dataset directory.

    Args:
        dataset_path (Path): The path to the dataset directory.

    Returns:
        dict: The metadata for the dataset.

    """
    with (dataset_path / "meta.json").open() as f:
        return json.load(f)


def add_xray_metadata(dataset_path: Path, xray_path: Path) -> None:
    """Add metadata about a sample X-ray image to the metadata dictionary.

    Args:
        dataset_path (Path): The path to the dataset directory.
        xray_path (Path): The path to the X-ray image.

    Returns:
        dict: The updated metadata dictionary with the X-ray metadata.

    """
    metadata = get_dataset_metadata(dataset_path)

    xray = sitk.ReadImage(xray_path)
    metadata["size"] = xray.GetSize()
    metadata["spacing"] = xray.GetSpacing()
    metadata["origin"] = xray.GetOrigin()
    metadata["direction"] = xray.GetDirection()
    metadata["dtype"] = {
        "id": xray.GetPixelID(),
        "value": xray.GetPixelIDValue(),
        "string": xray.GetPixelIDTypeAsString(),
    }
    metadata["extra_metadata"] = {key: xray.GetMetaData(key) for key in xray.GetMetaDataKeys()}

    with (dataset_path / "meta.json").open("w") as f:
        json.dump(metadata, f)

"""Utility functions."""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def get_data_dir() -> Path:
    """Get the path to the data directory."""
    return Path(__file__).parents[1] / "data"


def get_xray_dir() -> Path:
    """Get the path to the X-ray directory."""
    return get_data_dir() / "xrays"


def get_ct_dir() -> Path:
    """Get the path to the CT directory."""
    return get_data_dir() / "ct_images"


def get_model_dir() -> Path:
    """Get the path to the models directory."""
    return Path(__file__).parents[1] / "models"


def get_config_dir() -> Path:
    """Get the path to the config directory."""
    return Path(__file__).parents[1] / "config_yamls"


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
    try:
        with (dataset_path / "meta.json").open() as f:
            metadata = json.load(f)
    except FileNotFoundError as e:
        msg = f"Dataset not found: {dataset_path}"
        raise FileNotFoundError(msg) from e
    for file in dataset_path.iterdir():
        if file.suffix != ".json":
            try:
                xray = sitk.ReadImage(file)
                metadata = add_xray_metadata(metadata, xray)
                break
            except (OSError, RuntimeError) as e:
                print(f"Error reading {file}: {e}")  # noqa: T201
                continue
    return metadata


def add_xray_metadata(metadata: dict, xray: sitk.Image) -> dict:
    """Add metadata about a sample X-ray image to the metadata dictionary.

    Args:
        metadata (dict): The metadata dictionary to add the X-ray metadata to.
        xray (sitk.Image): The sample X-ray image.

    Returns:
        dict: The updated metadata dictionary with the X-ray metadata.

    """
    if "size" not in metadata:
        metadata["size"] = xray.GetSize()
    if "spacing" not in metadata:
        metadata["spacing"] = xray.GetSpacing()
    if "origin" not in metadata:
        metadata["origin"] = xray.GetOrigin()
    if "direction" not in metadata:
        metadata["direction"] = xray.GetDirection()
    if "dtype" not in metadata:
        metadata["dtype"] = {
            "id": xray.GetPixelID(),
            "value": xray.GetPixelIDValue(),
            "string": xray.GetPixelIDTypeAsString(),
        }
    if "extra_metadata" not in metadata:
        metadata["extra_metadata"] = {key: xray.GetMetaData(key) for key in xray.GetMetaDataKeys()}
    return metadata


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Get the torch dtype from a string.

    Args:
        dtype_str (str): The string representation of the torch dtype.

    Returns:
        torch.dtype: The torch dtype.

    """
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtypes.get(dtype_str)
    if dtype is None:
        msg = f"Unknown dtype: {dtype_str}"
        raise ValueError(msg)
    return dtype

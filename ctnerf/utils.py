import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def get_data_dir() -> Path:
    """
    Returns the path to the 'data' directory. This is the directory where all datasets are
    stored.
    """
    return Path(__file__).parents[1] / "data"


def get_model_dir() -> Path:
    """
    Returns the path to the 'models' directory. This is the directory where all model
    checkpoints are stored.
    """
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
    """
    Reads the metadata for a given dataset from the 'meta.json' file in the dataset directory.

    Args:
        dataset_path (Path): The path to the dataset directory.

    Returns:
        dict: The metadata for the dataset.
    """
    with open(dataset_path / "meta.json", "r") as f:
        metadata = json.load(f)
    return metadata


def add_xray_metadata(dataset_path: Path, xray_path: Path) -> None:
    """
    Adds metadata about a sample X-ray image to the metadata dictionary.

    Args:
        metadata (dict): The metadata dictionary to add the X-ray metadata to.
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

    with open(dataset_path / "meta.json", "w") as f:
        json.dump(metadata, f)

"""Defines configurations used for training and inferencing with the CT-NeRF model."""

from _collections_abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import yaml

from ctnerf.utils import (
    get_ct_dir,
    get_dataset_metadata,
    get_dtype,
    get_model_dir,
    get_xray_dir,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the CT-NeRF model."""

    # Scaling
    attenuation_scaling_factor: float | None  # scaling factor to raise X-rays to the reciprocal of
    s: float | None  # scaling factor for X-ray intensities
    k: float | None  # offset for X-ray intensities

    # Training
    seed: int  # seed for model initialization and ray sampling
    learning_rate: float
    batch_size: int  # batch size
    dtypes: dict[str, jax.numpy.dtype]  # data type of params, compute, input, and output
    checkpoint_dir: Path  # directory to save checkpoints
    checkpoint_interval: int  # interval to save checkpoints
    resume_training: bool  # whether or not this training is a resumption of an earlier one
    xray_dir: Path  # directory containing X-ray images
    num_workers: int  # number of dataloader workers

    # Model hyperparameters
    model: dict  # contains L, n_layers, layer_dim for both coarse and fine

    # Coarse model
    n_coarse_samples: int  # number of coarse samples
    plateau_ratio: float | None  # ratio of plateau width to standard deviation
    coarse_sampling_function: str  # name of coarse sampling function

    # Fine model
    n_fine_samples: int | None  # number of fine samples
    fine_sampling_function: str | None  # name of fine sampling function

    # Evaluation
    ct_size: tuple[int, int, int]  # size of the CT image to create for evaluation
    slice_size_cm: float  # size of an axial slice in centimetres
    source_ct_path: Path | None  # path to the source CT image

    # Documentation
    conf_dict: dict  # The original conf dict read directly from yaml


def get_training_config(config_path: Path) -> TrainingConfig:
    """Get the configuration for the CT-NeRF model.

    Loads the configuration from a YAML file and returns a TrainingConfig object.

    The inference yaml config file contains the following fields:

    - name: str. Name of the run

    - model:
        n_layers: int. Number of layers in the model
        layer_dim: int. Dimension of the layers
        L: int. Number of frequencies to use for the positional encoding

    - device: str. Device to run the model on

    - data:
        xray_dir: str. Directory containing the X-ray images
        source_ct_path: str. Path to the source CT image. Can be None
        num_workers: int. Number of workers to use for data loading

    - checkpoint:
        checkpoint_dir: str. Directory to load the model checkpoint from. If None, a new training
          run is created.
        checkpoint_interval: int. Interval to save checkpoints
        resume_epoch: int. Epoch to load the model from

    - training:
        seed: int. Seed for mode initialization and ray sampling
        lr: float. Learning rate
        batch_size: int. Batch size
        num_coarse_samples: int. Number of coarse samples
        coarse_sampling_function: str. Name of coarse sampling function, defined in ray_sampling.py
        num_fine_samples: int. Number of fine samples. If 0, no fine model is used

    - scaling:
        attenuation_scaling_factor: float | None. Scaling factor to raise X-ray to the reciprocal of
        s: float | None. Scaling factor
        k: float | None. Offset

    - dtypes:
        param_dtype: str. Should be float32, float16, or bfloat16.
        compute_dtype: str. Should be float32, float16, or bfloat16.
        output_dtype: str. Should be float32, float16, or bfloat16.


    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        TrainingConfig: The configuration for the CT-NeRF model.

    """
    # Load yaml config file
    with config_path.open("r") as f:
        conf_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Create checkpoint directory
    if conf_dict["checkpoint"].get("checkpoint_dir") is not None:
        checkpoint_dir = get_model_dir() / conf_dict["checkpoint"]["checkpoint_dir"]
        resumed_training = True
    else:
        checkpoint_dir = (
            get_model_dir() / conf_dict["name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resumed_training = False

    # Get X-rays and metadata
    xray_dir = get_xray_dir() / conf_dict["data"]["xray_dir"]
    metadata = get_dataset_metadata(xray_dir)
    slice_size_cm = metadata["spacing"][0] * metadata["size"][0] / 10
    ct_size = tuple([metadata["size"][0]] + metadata["size"])

    # Get dtypes
    allowed_dtypes = ("float32", "float16", "bfloat16")
    for value in conf_dict["dtypes"].values():
        if value not in allowed_dtypes:
            msg = f"Unknown dtype: {value}"
            raise ValueError(msg)
    required_keys = ("compute_dtype", "param_dtype", "input_dtype", "output_dtype")
    for key in required_keys:
        if key not in conf_dict["dtypes"]:
            msg = f"Missing dtype specification: {key}"
            raise ValueError(msg)
    dtypes = {key: get_dtype(item) for key, item in conf_dict["dtypes"].items()}

    return TrainingConfig(
        conf_dict=conf_dict,
        num_workers=conf_dict["data"]["num_workers"],
        learning_rate=conf_dict["training"]["lr"],
        attenuation_scaling_factor=conf_dict["scaling"].get("attenuation_scaling_factor"),
        s=conf_dict["scaling"].get("s"),
        k=conf_dict["scaling"].get("k"),
        seed=conf_dict["training"]["seed"],
        batch_size=conf_dict["training"]["batch_size"],
        dtypes=dtypes,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=conf_dict["checkpoint"]["checkpoint_interval"],
        resume_training=resumed_training,
        xray_dir=xray_dir,
        model=conf_dict["model"],
        n_coarse_samples=conf_dict["training"]["num_coarse_samples"],
        coarse_sampling_function=conf_dict["training"]["coarse_sampling_function"],
        fine_sampling_function=conf_dict["training"].get("fine_sampling_function", ""),
        plateau_ratio=conf_dict["training"].get("plateau_ratio"),
        n_fine_samples=conf_dict["training"].get("num_fine_samples"),
        ct_size=ct_size,
        slice_size_cm=slice_size_cm,
        source_ct_path=get_ct_dir() / conf_dict["data"]["source_ct_path"],
    )


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for the CT-NeRF model."""

    attenuation_scaling_factor: float | None  # scaling factor to raise X-rays to the reciprocal of
    output_path: Path  # path to save the generated CT image
    model: dict  # contains L, n_layers, layer_dim
    checkpoint_dir: Path  # path to saved model params
    image_size: list[int, int, int] | None  # size of the output image
    voxel_spacing: list[float, float, float] | None  # voxel spacing of the output image
    image_origin: list[float, float, float] | None  # origin of the output image
    image_direction: list[float] | None  # direction of the output image
    chunk_size: int  # number of rays to process in each batch
    xray_metadata: dict  # metadata of the input X-rays


def get_inference_config(config_path: Path) -> InferenceConfig:
    """Get the CT-NeRF inference configuration.

    The inference configuration is loaded from the specified YAML configuration file. Either
    img_size or spacing must be specified. If both are specified, spacing takes precedence.

    The inference yaml config file contains the following fields:

    - scaling:
        attenuation_scaling_factor: float. Scaling factor to raise X-ray images to the reciprocal of

    - model_type: 'coarse' or 'fine'

    - model:
        n_layers: int. Number of layers in the model
        layer_dim: int. Dimension of the layers
        L: int. Number of frequencies to use for the positional encoding

    - checkpoint:
        checkpoint_dir: str. Directory to load the model checkpoint from

    - output_dir: str. Directory to save the generated CT image

    - output_name: str. Name of the generated CT image

    - image_size: list[int, int, int]. Size of the output image. If None, spacing is used to
      determine the size. If both are specified, spacing takes precedence.

    - voxel_spacing: list[float, float, float]. Voxel spacing of the output image. If None,
      image_size is used to determine the spacing. If both are specified, spacing takes precedence.

    - image_origin: list[float, float, float]. Origin of the output image in sitk notation. If
      None, (0, 0, 0) is used.

    - image_direction: list[float]. Direction of the output image in sitk notation. If None,
      (1, 0, 0, 0, 1, 0, 0, 0, 1) is used.

    - chunk_size: int. Number of rays to process in each batch to avoid OOM errors. If None,
      4096 * 16 is used.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        InferenceConfig: The CT-NeRF inference configuration.

    """
    # Load yaml config file
    with config_path.open("r") as f:
        conf_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Get checkpoint dir
    checkpoint_dir = get_model_dir() / conf_dict["checkpoint"]["checkpoint_dir"]

    # Get X-ray metadata
    if "xray_dir" in conf_dict:
        xray_metadata = get_dataset_metadata(get_xray_dir() / conf_dict["xray_dir"])
    else:
        xray_metadata = {
            "size": conf_dict["xray_size"],
            "spacing": conf_dict["xray_pixel_spacing"],
        }
    if conf_dict.get("image_size") is None and conf_dict.get("voxel_spacing") is None:
        msg = "Either image_size or voxel_spacing must be specified."
        raise ValueError(msg)

    # Create output directory
    output_dir = get_ct_dir() / conf_dict["output_dir"]
    output_dir.mkdir(exist_ok=True, parents=True)

    return InferenceConfig(
        attenuation_scaling_factor=conf_dict["scaling"].get("attenuation_scaling_factor"),
        output_path=output_dir / conf_dict["output_name"],
        checkpoint_dir=checkpoint_dir,
        model=conf_dict["model"],
        chunk_size=conf_dict.get("chunk_size") or 4096 * 16,
        image_size=conf_dict.get("image_size"),
        voxel_spacing=conf_dict.get("voxel_spacing"),
        image_origin=conf_dict.get("image_origin") or [0, 0, 0],
        image_direction=conf_dict.get("image_direction") or [1, 0, 0, 0, 1, 0, 0, 0, 1],
        xray_metadata=conf_dict.get("xray_metadata") or xray_metadata,
    )

"""Defines configurations used for training and inferencing with the CT-NeRF model."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import yaml
from aim import Run
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from ctnerf.model import XRayModel
from ctnerf.utils import (
    get_ct_dir,
    get_dataset_metadata,
    get_model_dir,
    get_torch_dtype,
    get_xray_dir,
)

from .setup_functions import get_aim_run, get_dataloader, get_model, get_optimizer, load_checkpoint


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the CT-NeRF model."""

    # Scaling
    s: float  # scaling factor for X-ray intensities
    k: float  # offset for X-ray intensities

    # Training
    dataloader: DataLoader  # data loader
    batch_size: int  # batch size
    loss_fn: torch.nn.Module  # loss function
    use_amp: bool  # use automatic mixed precision
    device: torch.device  # device to run on
    dtype: torch.dtype  # data type of input tensors
    checkpoint_dir: Path  # directory to save checkpoints
    checkpoint_interval: int  # interval to save checkpoints
    xray_dir: Path  # directory containing X-ray images
    start_epoch: int  # starting epoch
    tracker: Run  # aim tracker

    # Coarse model
    coarse_model: XRayModel | None  # coarse model
    coarse_optimizer: torch.optim.Optimizer | None  # coarse optimizer
    coarse_scaler: torch.GradScaler | None  # coarse gradient scaler
    n_coarse_samples: int  # number of coarse samples
    plateau_ratio: float  # ratio of plateau width to standard deviation

    # Fine model
    fine_model: XRayModel | None  # fine model
    fine_optimizer: torch.optim.Optimizer | None  # fine optimizer
    fine_scaler: torch.GradScaler | None  # fine gradient scaler
    n_fine_samples: int | None  # number of fine samples

    # Evaluation
    ct_size: tuple[int, int, int]  # size of the CT image to create for evaluation
    slice_size_cm: float  # size of an axial slice in centimetres
    source_ct_path: Path | None  # path to the source CT image


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
        pin_memory: bool. Whether to pin memory for data loading

    - checkpoint:
        checkpoint_dir: str. Directory to load the model checkpoint from. If None, a new training
          run is created.
        checkpoint_interval: int. Interval to save checkpoints
        resume_epoch: int. Epoch to load the model from

    - training:
        lr: float. Learning rate
        batch_size: int. Batch size
        num_coarse_samples: int. Number of coarse samples
        num_fine_samples: int. Number of fine samples. If None, no fine model is used
        dtype: str. Data type of the input tensors
        use_amp: bool. Whether to use automatic mixed precision

    - scaling:
        s: float. Scaling factor
        k: float. Offset

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        TrainingConfig: The configuration for the CT-NeRF model.

    """
    # Load yaml config file
    with config_path.open("r") as f:
        conf_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Get coarse model
    coarse_model = get_model(conf_dict)
    coarse_optimizer = get_optimizer(conf_dict, coarse_model)
    coarse_scaler = torch.GradScaler()

    # Get fine model if specified
    if conf_dict["training"].get("num_fine_samples") is not None:
        fine_model = get_model(conf_dict)
        fine_optimizer = get_optimizer(conf_dict, fine_model)
        fine_scaler = torch.GradScaler()
    else:
        fine_model = None
        fine_optimizer = None
        fine_scaler = None

    # Load checkpoint. If checkpoint_dir is not specified, this will be a no-op.
    start_epoch, run_hash = load_checkpoint(
        conf_dict,
        coarse_model,
        coarse_optimizer,
        fine_model,
        fine_optimizer,
    )
    run = get_aim_run(conf_dict, run_hash)

    # Create checkpoint directory
    if conf_dict["checkpoint"].get("checkpoint_dir") is not None:
        checkpoint_dir = get_model_dir() / conf_dict["checkpoint"]["checkpoint_dir"]
    else:
        checkpoint_dir = (
            get_model_dir() / conf_dict["name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloader and loss function
    dataloader = get_dataloader(conf_dict)
    loss_fn = MSELoss(reduction="none")

    # Get X-rays and metadata
    xray_dir = get_xray_dir() / conf_dict["data"]["xray_dir"]
    metadata = get_dataset_metadata(xray_dir)
    slice_size_cm = metadata["spacing"][0] * metadata["size"][0] / 10
    ct_size = [metadata["size"][0]] + metadata["size"]

    return TrainingConfig(
        s=conf_dict["scaling"]["s"],
        k=conf_dict["scaling"]["k"],
        dataloader=dataloader,
        batch_size=conf_dict["training"]["batch_size"],
        loss_fn=loss_fn,
        use_amp=conf_dict["training"]["use_amp"],
        device=torch.device(conf_dict["device"]),
        dtype=get_torch_dtype(conf_dict["training"]["dtype"]),
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=conf_dict["checkpoint"]["checkpoint_interval"],
        xray_dir=xray_dir,
        start_epoch=start_epoch,
        tracker=run,
        coarse_model=coarse_model,
        coarse_optimizer=coarse_optimizer,
        coarse_scaler=coarse_scaler,
        n_coarse_samples=conf_dict["training"]["num_coarse_samples"],
        plateau_ratio=conf_dict["training"]["plateau_ratio"],
        fine_model=fine_model,
        fine_optimizer=fine_optimizer,
        fine_scaler=fine_scaler,
        n_fine_samples=conf_dict["training"].get("num_fine_samples"),
        ct_size=ct_size,
        slice_size_cm=slice_size_cm,
        source_ct_path=get_ct_dir() / conf_dict["data"]["source_ct_path"],
    )


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for the CT-NeRF model."""

    coarse_model: XRayModel | None  # coarse model
    fine_model: XRayModel | None  # fine model
    output_path: Path  # path to save the generated CT image
    device: torch.device  # device to run the model inference on
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

    - model_type: 'coarse' or 'fine'

    - model:
        n_layers: int. Number of layers in the model
        layer_dim: int. Dimension of the layers
        L: int. Number of frequencies to use for the positional encoding

    - device: str. Device to run the model inference on

    - checkpoint:
        checkpoint_dir: str. Directory to load the model checkpoint from
        resume_epoch: int. Epoch to load the model from

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

    # Get model and load checkpoint
    if conf_dict["model_type"] == "coarse":
        coarse_model = get_model(conf_dict)
        fine_model = None
    elif conf_dict["model_type"] == "fine":
        coarse_model = None
        fine_model = get_model(conf_dict)
    else:
        msg = f"Unknown model type: {conf_dict['model']}"
        raise ValueError(msg)
    load_checkpoint(conf_dict, coarse_model=coarse_model, fine_model=fine_model)

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
        coarse_model=coarse_model,
        fine_model=fine_model,
        output_path=output_dir / conf_dict["output_name"],
        chunk_size=conf_dict.get("chunk_size") or 4096 * 16,
        device=conf_dict.get("device") or torch.device("cpu"),
        image_size=conf_dict.get("image_size"),
        voxel_spacing=conf_dict.get("voxel_spacing"),
        image_origin=conf_dict.get("image_origin") or [0, 0, 0],
        image_direction=conf_dict.get("image_direction") or [1, 0, 0, 0, 1, 0, 0, 0, 1],
        xray_metadata=conf_dict.get("xray_metadata") or xray_metadata,
    )

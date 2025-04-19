"""Functions for setting up the CT-NeRF model."""

from pathlib import Path

import torch
from aim import Run

from ctnerf.model import XRayModel
from ctnerf.training.dataloading import XRayDataset
from ctnerf.utils import get_model_dir, get_torch_dtype, get_xray_dir


def get_model(conf_dict: dict) -> XRayModel:
    """Get the CT-NeRF model and send it to the specified device.

    Args:
        conf_dict (dict): The configuration dictionary.

    Returns:
        (XRayModel): The CT-NeRF model.

    """
    return XRayModel(
        n_layers=conf_dict["model"]["n_layers"],
        layer_dim=conf_dict["model"]["layer_dim"],
        L=conf_dict["model"]["L"],
    ).to(conf_dict["device"])


def get_optimizer(conf_dict: dict, model: XRayModel) -> torch.optim.Optimizer:
    """Get the optimizer for the specified model.

    Args:
        conf_dict (dict): The configuration dictionary.
        model (XRayModel): The CT-NeRF model.

    Returns:
        (torch.optim.Optimizer): The optimizer.

    """
    return torch.optim.Adam(model.parameters(), fused=True, lr=conf_dict["training"]["lr"])


def load_checkpoint(
    conf_dict: dict,
    coarse_model: XRayModel | None = None,
    coarse_optimizer: torch.optim.Optimizer | None = None,
    fine_model: XRayModel | None = None,
    fine_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, str]:
    """Load the checkpoint if it exists.

    Args:
        conf_dict (dict): The configuration dictionary.
        coarse_model (XRayModel, optional): The coarse model. Defaults to None.
        coarse_optimizer (torch.optim.Optimizer, optional): The coarse optimizer. Defaults to None.
        fine_model (XRayModel, optional): The fine model. Defaults to None.
        fine_optimizer (torch.optim.Optimizer, optional): The fine optimizer. Defaults to None.

    Returns:
        tuple[int, int, str]: The epoch and run hash of the checkpoint if it exists, else (0, "").

    """
    if conf_dict["checkpoint"].get("checkpoint_dir") is not None:
        checkpoint_path = Path(
            get_model_dir() / conf_dict["checkpoint"]["checkpoint_dir"],
            (str(conf_dict["checkpoint"]["resume_epoch"]) + ".pt"),
        )
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=conf_dict["device"],
        )
        if coarse_model is not None:
            coarse_model.load_state_dict(checkpoint["coarse_model_state_dict"])
        if coarse_optimizer is not None:
            coarse_optimizer.load_state_dict(checkpoint["coarse_optimizer_state_dict"])
        if fine_model is not None:
            fine_model.load_state_dict(checkpoint["fine_model_state_dict"])
        if fine_optimizer is not None:
            fine_optimizer.load_state_dict(checkpoint["fine_optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["run_hash"]
    return 0, ""


def get_dataloader(conf_dict: dict) -> torch.utils.data.DataLoader:
    """Get the data loader for the specified configuration.

    Args:
        conf_dict (dict): The configuration dictionary.

    Returns:
        (torch.utils.data.DataLoader): The data loader.

    """
    dataset = XRayDataset(
        xray_dir=get_xray_dir() / conf_dict["data"]["xray_dir"],
        dtype=get_torch_dtype(conf_dict["training"]["dtype"]),
        attenuation_scaling_factor=conf_dict["scaling"].get("attenuation_scaling_factor"),
        s=conf_dict["scaling"].get("s"),
        k=conf_dict["scaling"].get("k"),
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=conf_dict["training"]["batch_size"],
        shuffle=True,
        num_workers=conf_dict["data"]["num_workers"],
        pin_memory=conf_dict["data"]["pin_memory"],
        pin_memory_device=conf_dict["device"],
    )


def get_aim_run(conf_dict: dict, run_hash: str) -> Run:
    """Get the Aim run for the specified configuration.

    Args:
        conf_dict (dict): The configuration dictionary.
        run_hash (str): The hash of the run.

    Returns:
        (Run): The Aim run.

    """
    run = Run(log_system_params=True) if run_hash == "" else Run(run_hash, log_system_params=True)
    run["hparams"] = conf_dict
    return run

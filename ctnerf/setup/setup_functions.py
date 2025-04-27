"""Functions for setting up the CT-NeRF model."""

import jax
import numpy as np
import optax
from aim import Run
from jax.tree_util import tree_map
from torch.utils import data
from torch.utils.data import DataLoader

from ctnerf.model import init_params
from ctnerf.setup.config import TrainingConfig
from ctnerf.training.dataloading import XrayDataset
from ctnerf.utils import get_xray_dir


def get_model(
    conf: TrainingConfig,
) -> tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]]:
    """Get the CT-NeRF model and send it to the specified device.

    Args:
        conf (TrainingConfig): The configuration dataclass.

    Returns:
        (tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]]): The CT-NeRF model.

    """
    return init_params(
        key=jax.random.key(conf.model["seed"]),
        n_layers=conf.model["n_layers"],
        layer_dim=conf.model["layer_dim"],
        L=conf.model["L"],
    )


def get_optimizer(
    conf: TrainingConfig,
    model: tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]]],
) -> tuple[optax.GradientTransformation, optax.OptState]:
    """Get the optimizer for the specified model.

    Args:
        conf (TrainingConfig): The configuration dataclass.
        model (XRayModel): The CT-NeRF model.

    Returns:
        (torch.optim.Optimizer): The optimizer.

    """
    optimizer = optax.adam(learning_rate=conf.learning_rate)
    opt_state = optimizer.init(model)
    return optimizer, opt_state


def get_dataloader(conf: TrainingConfig) -> DataLoader:
    """Get the data loader for the specified configuration.

    Args:
        conf (TrainingConfig): The configuration dataclass.

    Returns:
        (DataLoader): The data loader.

    """
    dataset = XrayDataset(
        xray_dir=get_xray_dir() / conf.xray_dir,
        dtype=conf.dtype,
        attenuation_scaling_factor=conf.attenuation_scaling_factor,
        s=conf.s,
        k=conf.k,
    )

    def numpy_collate(
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tree_map(np.asarray, data.default_collate(batch))

    return DataLoader(
        dataset=dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        collate_fn=numpy_collate,
        drop_last=True,
        prefetch_factor=4,
    )


def get_aim_run(conf: TrainingConfig, run_hash: str) -> Run:
    """Get the Aim run for the specified configuration.

    Args:
        conf (TrainingConfig): The configuration dataclass.
        run_hash (str): The hash of the run.

    Returns:
        (Run): The Aim run.

    """
    run = Run(log_system_params=True) if run_hash == "" else Run(run_hash, log_system_params=True)
    run["hparams"] = conf.conf_dict
    return run

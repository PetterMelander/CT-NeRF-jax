"""Contains functions for training the model."""

from pathlib import Path

import numpy as np
import plotly.express as px
import SimpleITK as sitk
import torch
from aim import Figure
from torch import GradScaler, autocast
from tqdm import tqdm

from ctnerf.image_creation.ct_creation import run_inference, tensor_to_sitk
from ctnerf.model import XRayModel
from ctnerf.rays import get_coarse_samples, get_fine_samples, log_beer_lambert_law
from ctnerf.setup.config import TrainingConfig, get_training_config


def train(config_path: Path) -> None:
    """Train the model."""
    conf = get_training_config(config_path)

    for epoch in range(1, 1000):
        for start_positions, heading_vectors, intensities, ray_bounds in tqdm(conf.dataloader):
            _step(
                start_positions,
                heading_vectors,
                intensities,
                ray_bounds,
                conf,
            )

        _eval(
            conf.coarse_model,
            conf.start_epoch + epoch,
            "coarse",
            conf,
        )

        if conf.fine_model is not None:
            _eval(
                conf.fine_model,
                conf.start_epoch + epoch,
                "fine",
                conf,
            )

        if epoch % conf.checkpoint_interval == 0:
            checkpoint = {
                "coarse_model_state_dict": conf.coarse_model.state_dict(),
                "coarse_optimizer_state_dict": conf.coarse_optimizer.state_dict(),
                "epoch": conf.start_epoch + epoch,
                "run_hash": conf.tracker.hash,
            }
            if conf.fine_model is not None:
                checkpoint["fine_model_state_dict"] = conf.fine_model.state_dict()
                checkpoint["fine_optimizer_state_dict"] = conf.fine_optimizer.state_dict()
            torch.save(checkpoint, conf.checkpoint_dir / f"{conf.start_epoch + epoch}.pt")


@torch.compile(mode="max-autotune", disable=False)
def _step(
    start_positions: torch.Tensor,
    heading_vectors: torch.Tensor,
    intensities: torch.Tensor,
    ray_bounds: torch.Tensor,
    conf: TrainingConfig,
) -> None:
    start_positions = start_positions.to(conf.device, dtype=conf.dtype, non_blocking=True)
    heading_vectors = heading_vectors.to(conf.device, dtype=conf.dtype, non_blocking=True)
    intensities = intensities.to(conf.device, dtype=conf.dtype, non_blocking=True)
    ray_bounds = ray_bounds.to(conf.device, dtype=conf.dtype, non_blocking=True)

    (
        coarse_sample_ts,
        coarse_attenuation_coeff_pred,
        coarse_sampling_distances,
    ) = _coarse_step(
        start_positions,
        heading_vectors,
        intensities,
        ray_bounds,
        conf,
    )

    if conf.fine_model is not None:
        _fine_step(
            start_positions,
            heading_vectors,
            intensities,
            ray_bounds,
            coarse_sample_ts,
            coarse_attenuation_coeff_pred,
            coarse_sampling_distances,
            conf,
        )


# @torch.compile(mode="max-autotune", disable=True)
def _coarse_step(
    start_positions: torch.Tensor,
    heading_vectors: torch.Tensor,
    intensities: torch.Tensor,
    ray_bounds: torch.Tensor,
    conf: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coarse_sample_ts, coarse_samples, coarse_sampling_distances = get_coarse_samples(
        start_positions,
        heading_vectors,
        ray_bounds,
        conf.n_coarse_samples,
    )

    loss, attenuation_coeff_pred = _forward_backward(
        intensities,
        coarse_samples,
        coarse_sampling_distances,
        conf.coarse_model,
        conf.coarse_optimizer,
        conf.coarse_scaler,
        conf.loss_fn,
        conf,
    )

    conf.tracker.track(loss.item(), name="coarse_loss")

    return (
        coarse_sample_ts.detach(),
        attenuation_coeff_pred.detach(),
        coarse_sampling_distances.detach(),
    )


# @torch.compile(mode="max-autotune", disable=True)
def _fine_step(
    start_positions: torch.Tensor,
    heading_vectors: torch.Tensor,
    intensities: torch.Tensor,
    ray_bounds: torch.Tensor,
    coarse_sample_ts: torch.Tensor,
    attenuation_coeff_pred: torch.Tensor,
    coarse_sampling_distances: torch.Tensor,
    conf: TrainingConfig,
) -> None:
    fine_samples, fine_sampling_distances = get_fine_samples(
        start_positions,
        heading_vectors,
        ray_bounds,
        coarse_sample_ts,
        attenuation_coeff_pred,
        coarse_sampling_distances,
        conf.n_fine_samples,
    )

    loss, _ = _forward_backward(
        intensities,
        fine_samples,
        fine_sampling_distances,
        conf.fine_model,
        conf.fine_optimizer,
        conf.fine_scaler,
        conf.loss_fn,
        conf,
    )

    conf.tracker.track(loss.item(), name="loss")


def _forward_backward(
    intensities: torch.Tensor,
    samples: torch.Tensor,
    sampling_distances: torch.Tensor,
    model: XRayModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: torch.nn.Module,
    conf: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()

    with autocast(device_type="cuda", enabled=conf.use_amp):
        attenuation_coeff_pred = model(samples)
        attenuation_coeff_pred = attenuation_coeff_pred.reshape(conf.batch_size, -1)
        intensity_pred = log_beer_lambert_law(
            attenuation_coeff_pred,
            sampling_distances,
            conf.s,
            conf.k,
            conf.slice_size_cm,
        )
        loss = loss_fn(intensity_pred, intensities)
        loss = torch.sum(loss)

    if conf.use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    return loss, attenuation_coeff_pred


@torch.no_grad()
def _eval(
    model: torch.nn.Module,
    epoch: int,
    tag: str,
    conf: TrainingConfig,
) -> None:
    model.eval()

    if conf.source_ct_path is not None:
        generated_ct = run_inference(model, conf.ct_size, 4096 * 64, conf.device)
        ct_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        generated_ct = tensor_to_sitk(generated_ct, direction=ct_direction)
        generated_ct = sitk.GetArrayFromImage(generated_ct)

        source_ct_image = sitk.ReadImage(str(conf.source_ct_path))
        source_ct_image = sitk.GetArrayFromImage(source_ct_image)

        mae = np.mean(np.abs(generated_ct - source_ct_image))
        conf.tracker.track(mae, name="mae", step=epoch, context={"model": tag})

        del generated_ct
        del source_ct_image

    yv = torch.linspace(-1, 1, conf.ct_size[1])
    zv = torch.linspace(-1, 1, conf.ct_size[2])
    yv, zv = torch.meshgrid(yv, zv, indexing="xy")
    coords = torch.stack([torch.zeros_like(yv), yv, zv], dim=-1)
    coords = coords.reshape(-1, 3)
    coords = coords.to(conf.device)

    output = model(coords)
    output = output.reshape(conf.ct_size[2], conf.ct_size[1])

    fig = px.imshow(output.cpu().numpy(), color_continuous_scale="gray")
    conf.tracker.track(Figure(fig), name="cross section", step=epoch, context={"model": tag})

    del output
    del coords

    model.train()

"""Script for training the model. Currently very ugly."""

import datetime
from pathlib import Path

import numpy as np
import plotly.express as px
import SimpleITK as sitk
import torch
from aim import Figure, Run
from torch import GradScaler, autocast
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ctnerf.constants import MU_AIR, MU_WATER
from ctnerf.dataloading import XRayDataset
from ctnerf.model import XRayModel
from ctnerf.rays import get_coarse_samples, get_fine_samples, log_beer_lambert_law
from ctnerf.utils import get_data_dir, get_dataset_metadata, get_model_dir

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.allow_tf32 = True
torch.backends.cuda.preferred_blas_library("cublaslt")


def train() -> None:  # noqa: PLR0915
    """Train the model."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    hparams = {
        "name": "coarse-only",
        "dataset": "test",
        "source_ct_image_path": str(get_data_dir() / "ct_images" / "nrrd" / "2 AC_CT_TBody.nrrd"),
        # "source_ct_image_path": None,  # noqa: ERA001
        "model_save_interval": 1,
        # "model_load_path": f"{get_model_dir()}/coarse-only/20241017-174113",  # noqa: ERA001
        "model_load_path": None,
        "resume_epoch": 1,
        "device": "cuda:0",
        "model": {
            "n_layers": 8,
            "layer_dim": 256,
            "L": 20,
        },
        "training": {
            "lr": 0.0001,
            "batch_size": 4096,
            "num_coarse_samples": 64,
            "num_fine_samples": 128,
            "dtype": "float32",
            "use_amp": True,
        },
        "scaling": {
            "s": 1,
            "k": 0.1,
        },
    }

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtypes.get(hparams["training"]["dtype"])
    if dtype is None:
        msg = f"Unknown dtype: {hparams['dtype']}"
        raise ValueError(msg)

    checkpoint_path = get_model_dir() / f"{hparams['name']}" / f"{timestamp}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    xray_dir = get_data_dir() / "xrays" / hparams["dataset"]
    dataset = XRayDataset(xray_dir, dtype=dtype, **hparams["scaling"])
    dataloader = DataLoader(
        dataset,
        batch_size=hparams["training"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        pin_memory_device=hparams["device"],
    )

    metadata = get_dataset_metadata(xray_dir)
    hparams["scaling"]["slice_size_cm"] = metadata["spacing"][0] * metadata["size"][0] / 10
    hparams["ct_size"] = [metadata["size"][0]] + metadata["size"]

    coarse_model = XRayModel(**hparams["model"])
    fine_model = XRayModel(**hparams["model"])
    coarse_model.to(hparams["device"])
    fine_model.to(hparams["device"])

    coarse_optimizer = Adam(coarse_model.parameters(), fused=True, lr=hparams["training"]["lr"])
    fine_optimizer = Adam(fine_model.parameters(), fused=True, lr=hparams["training"]["lr"])

    if hparams["model_load_path"] is not None:
        model_file = Path(hparams["model_load_path"]) / f"{hparams['resume_epoch']}.pt"
        checkpoint = torch.load(model_file, weights_only=True, map_location=hparams["device"])
        coarse_model.load_state_dict(checkpoint["coarse_model"])
        fine_model.load_state_dict(checkpoint["fine_model"])
        coarse_optimizer.load_state_dict(checkpoint["coarse_optimizer"])
        fine_optimizer.load_state_dict(checkpoint["fine_optimizer"])
        total_batches = checkpoint["total_batches"]
        start_epoch = checkpoint["epoch"]
        run_hash = checkpoint["run_hash"]
    else:
        total_batches = 0
        start_epoch = 0
        run_hash = None

    run = Run(run_hash)
    run["hparams"] = hparams

    mse_loss = MSELoss(reduction="none")

    scaler_coarse = GradScaler()
    scaler_fine = GradScaler()

    for epoch in range(1, 1000):
        for start_positions, heading_vectors, intensities, ray_bounds in tqdm(dataloader):
            start_positions = start_positions.to(hparams["device"], dtype=dtype, non_blocking=True)
            heading_vectors = heading_vectors.to(hparams["device"], dtype=dtype, non_blocking=True)
            intensities = intensities.to(hparams["device"], dtype=dtype, non_blocking=True)
            ray_bounds = ray_bounds.to(hparams["device"], dtype=dtype, non_blocking=True)

            (
                coarse_sample_ts,
                coarse_attenuation_coeff_pred,
                coarse_sampling_distances,
                coarse_loss,
            ) = _coarse_step(
                start_positions,
                heading_vectors,
                intensities,
                ray_bounds,
                coarse_model,
                coarse_optimizer,
                scaler_coarse,
                mse_loss,
                hparams,
            )

            loss = _fine_step(
                start_positions,
                heading_vectors,
                intensities,
                ray_bounds,
                fine_model,
                fine_optimizer,
                scaler_fine,
                mse_loss,
                coarse_sample_ts,
                coarse_attenuation_coeff_pred,
                coarse_sampling_distances,
                hparams,
            )
            total_batches += 1
            run.track(loss.item(), name="loss", step=total_batches)
            run.track(coarse_loss.item(), name="coarse_loss", step=total_batches)

        _eval(
            coarse_model,
            hparams["ct_size"],
            hparams["source_ct_image_path"],
            hparams["device"],
            start_epoch + epoch,
            run,
            "coarse",
        )
        _eval(
            fine_model,
            hparams["ct_size"],
            hparams["source_ct_image_path"],
            hparams["device"],
            start_epoch + epoch,
            run,
            "fine",
        )

        if epoch % hparams["model_save_interval"] == 0:
            torch.save(
                {
                    "coarse_model": coarse_model.state_dict(),
                    "fine_model": fine_model.state_dict(),
                    "coarse_optimizer": coarse_optimizer.state_dict(),
                    "fine_optimizer": fine_optimizer.state_dict(),
                    "total_batches": (start_epoch + epoch) * len(dataloader),
                    "epoch": start_epoch + epoch,
                    "run_hash": run.hash,
                    "model_hparams": hparams["model"],
                },
                checkpoint_path / f"{start_epoch + epoch}.pt",
            )


@torch.compile(mode="max-autotune", disable=False)
def _coarse_step(
    start_positions: torch.Tensor,
    heading_vectors: torch.Tensor,
    intensities: torch.Tensor,
    ray_bounds: torch.Tensor,
    coarse_model: torch.nn.Module,
    coarse_optimizer: torch.optim.Optimizer,
    scaler_coarse: GradScaler,
    loss_fn: torch.nn.Module,
    hparams: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coarse_sample_ts, coarse_samples, coarse_sampling_distances = get_coarse_samples(
        start_positions,
        heading_vectors,
        ray_bounds,
        hparams["training"]["num_coarse_samples"],
    )

    loss, attenuation_coeff_pred = _forward_backward(
        intensities,
        coarse_model,
        coarse_optimizer,
        scaler_coarse,
        loss_fn,
        coarse_samples,
        coarse_sampling_distances,
        hparams,
    )

    return (
        coarse_sample_ts.detach(),
        attenuation_coeff_pred.detach(),
        coarse_sampling_distances.detach(),
        loss.detach(),
    )


@torch.compile(mode="max-autotune", disable=False)
def _fine_step(
    start_positions: torch.Tensor,
    heading_vectors: torch.Tensor,
    intensities: torch.Tensor,
    ray_bounds: torch.Tensor,
    fine_model: XRayModel,
    fine_optimizer: torch.optim.Optimizer,
    scaler_fine: GradScaler,
    loss_fn: torch.nn.Module,
    coarse_sample_ts: torch.Tensor,
    attenuation_coeff_pred: torch.Tensor,
    coarse_sampling_distances: torch.Tensor,
    hparams: dict,
) -> torch.Tensor:
    fine_samples, fine_sampling_distances = get_fine_samples(
        start_positions,
        heading_vectors,
        ray_bounds,
        coarse_sample_ts,
        attenuation_coeff_pred,
        coarse_sampling_distances,
        hparams["training"]["num_fine_samples"],
    )

    loss, _ = _forward_backward(
        intensities,
        fine_model,
        fine_optimizer,
        scaler_fine,
        loss_fn,
        fine_samples,
        fine_sampling_distances,
        hparams,
    )

    return loss.detach()


def _forward_backward(
    intensities: torch.Tensor,
    model: XRayModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: torch.nn.Module,
    samples: torch.Tensor,
    sampling_distances: torch.Tensor,
    hparams: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()

    with autocast(device_type="cuda", enabled=hparams["training"]["use_amp"]):
        attenuation_coeff_pred = model(samples)
        attenuation_coeff_pred = attenuation_coeff_pred.reshape(
            hparams["training"]["batch_size"],
            -1,
        )
        intensity_pred = log_beer_lambert_law(
            attenuation_coeff_pred,
            sampling_distances,
            **hparams["scaling"],
        )
        loss = loss_fn(intensity_pred, intensities)
        loss = torch.sum(loss)

    if hparams["training"]["use_amp"]:
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
    img_size: tuple[int, int, int],
    source_ct_path: str | None,
    device: torch.device,
    epoch: int,
    run: Run,
    tag: str,
) -> None:
    model.eval()

    if source_ct_path is not None:
        x = torch.linspace(-1, 1, img_size[0])
        y = torch.linspace(-1, 1, img_size[1])
        z = torch.linspace(-1, 1, img_size[2])

        coords = torch.stack(torch.meshgrid((x, y, z), indexing="xy"), dim=-1)
        coords = coords.view(-1, 3)

        # To avoid oom, inference is done in batches and result stored on cpu
        output = torch.empty(coords.shape[0], device="cpu")
        coords = coords.split(4096 * 16, dim=0)
        index = 0
        for chunk in tqdm(coords, desc="Generating", total=len(coords)):
            chunk = chunk.to(device)
            output_chunk = model(chunk)
            output_chunk = output_chunk.view(-1)
            output[index : index + output_chunk.shape[0]] = output_chunk.cpu()
            index += output_chunk.shape[0]

        # Convert to hounsfield
        output = 1000 * (output - MU_WATER) / (MU_WATER - MU_AIR)

        output = output.reshape(img_size[0], img_size[1], img_size[2])
        output = torch.permute(output, (2, 1, 0))
        output = output.clamp(min=-1024)
        ct_image = sitk.GetImageFromArray(output.numpy().astype("int16"))
        ct_image.SetDirection([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        ct_image = sitk.GetArrayFromImage(ct_image)

        source_ct_image = sitk.ReadImage(str(source_ct_path))
        source_ct_image = sitk.GetArrayFromImage(source_ct_image)

        mae = np.mean(np.abs(ct_image - source_ct_image))
        run.track(mae, name="mae", step=epoch, context={"model": tag})

        del output
        del source_ct_image
        del coords

    yv = torch.linspace(-1, 1, img_size[1])
    zv = torch.linspace(-1, 1, img_size[2])
    yv, zv = torch.meshgrid(yv, zv, indexing="xy")
    coords = torch.stack([torch.zeros_like(yv), yv, zv], dim=-1)
    coords = coords.reshape(-1, 3)
    coords = coords.to(device)

    output = model(coords)
    output = output.reshape(img_size[2], img_size[1])

    fig = px.imshow(output.cpu().numpy(), color_continuous_scale="gray")
    run.track(Figure(fig), name="cross section", step=epoch, context={"model": tag})

    del output
    del coords

    model.train()


if __name__ == "__main__":
    train()

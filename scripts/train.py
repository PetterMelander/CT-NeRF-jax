import datetime
from pathlib import Path

import plotly.express as px
import torch
from aim import Figure, Run
from torch import GradScaler, autocast
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ctnerf.dataloading import XRayDataset
from ctnerf.models import XRayModel
from ctnerf.rays import get_coarse_samples, get_fine_samples, log_beer_lambert_law
from ctnerf.utils import get_data_dir, get_dataset_metadata, get_model_dir

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.allow_tf32 = True


def train():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    hparams = {
        "name": "coarse-only",
        "dataset": "test",
        "model_save_interval": 1,
        "model_load_path": f"{get_model_dir()}/coarse-only/20241017-174113",
        # "model_load_path": None,
        "resume_epoch": 71,
        "device": "cuda:0",
        "model": {
            "n_layers": 12,
            "layer_dim": 256,
            "L": 20,
        },
        "training": {
            "lr": 0.0001,
            "batch_size": 4096,
            "num_coarse_samples": 256,
            "num_fine_samples": 128,
            "dtype": "float32",
            "use_amp": False,
        },
        "scaling": {
            "s": 1,
            "k": 0.1,
        },
    }

    if hparams["training"]["dtype"] == "bfloat16":
        dtype = torch.bfloat16
    elif hparams["training"]["dtype"] == "float16":
        dtype = torch.float16
    elif hparams["training"]["dtype"] == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {hparams['dtype']}")

    model_save_path = get_model_dir() / f"{hparams['name']}" / f"{timestamp}"
    model_save_path.mkdir(parents=True, exist_ok=True)

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

    coarse_model = XRayModel(
        **hparams["model"],
    )
    # fine_model = XRayModel(
    #     **hparams["model"],
    # )
    coarse_model.to(hparams["device"])
    # fine_model.to(hparams["device"])

    coarse_optimizer = Adam(coarse_model.parameters(), fused=True, lr=hparams["training"]["lr"])
    # fine_optimizer = Adam(fine_model.parameters(), fused=True, lr=hparams["training"]["lr"])

    if hparams["model_load_path"] is not None:
        model_file = Path(hparams["model_load_path"]) / f"{hparams['resume_epoch']}.pt"
        weights = torch.load(model_file, weights_only=True, map_location=hparams["device"])
        coarse_model.load_state_dict(weights["coarse_model"])
        # fine_model.load_state_dict(weights["fine_model"])
        coarse_optimizer.load_state_dict(weights["coarse_optimizer"])
        # fine_optimizer.load_state_dict(weights["fine_optimizer"])
        total_batches = weights["total_batches"]
        start_epoch = weights["epoch"]
        run_hash = weights["run_hash"]
    else:
        total_batches = 0
        start_epoch = 0
        run_hash = None

    run = Run(run_hash)
    run["hparams"] = hparams

    mse_loss = MSELoss(reduction="none")

    scaler_coarse = GradScaler()
    # scaler_fine = GradScaler()

    for epoch in range(1, 1000):
        for start_positions, heading_vectors, intensities in tqdm(dataloader):
            start_positions = start_positions.to(hparams["device"], dtype=dtype, non_blocking=True)
            heading_vectors = heading_vectors.to(hparams["device"], dtype=dtype, non_blocking=True)
            intensities = intensities.to(hparams["device"], dtype=dtype, non_blocking=True)

            (
                coarse_sample_ts,
                coarse_attenuation_coeff_pred,
                coarse_sampling_distances,
                coarse_loss,
            ) = _coarse_step(
                start_positions,
                heading_vectors,
                intensities,
                coarse_model,
                coarse_optimizer,
                scaler_coarse,
                mse_loss,
                hparams,
            )

            # loss = _fine_step(
            #     start_positions,
            #     heading_vectors,
            #     intensities,
            #     fine_model,
            #     fine_optimizer,
            #     scaler_fine,
            #     mse_loss,
            #     coarse_sample_ts,
            #     coarse_attenuation_coeff_pred,
            #     coarse_sampling_distances,
            #     hparams,
            # )
            total_batches += 1
            # run.track(loss.cpu().item(), name="loss", step=total_batches)
            run.track(coarse_loss.cpu().item(), name="coarse_loss", step=total_batches)

        # generate cross section
        # _eval_fig(
        #     fine_model, hparams["ct_size"][1:], hparams["device"], start_epoch + epoch, run, "fine"
        # )
        _eval_fig(
            coarse_model,
            hparams["ct_size"][1:],
            hparams["device"],
            start_epoch + epoch,
            run,
            "coarse",
        )

        if epoch % hparams["model_save_interval"] == 0:
            torch.save(
                {
                    "coarse_model": coarse_model.state_dict(),
                    # "fine_model": fine_model.state_dict(),
                    "coarse_optimizer": coarse_optimizer.state_dict(),
                    # "fine_optimizer": fine_optimizer.state_dict(),
                    "total_batches": total_batches,
                    "epoch": start_epoch + epoch,
                    "run_hash": run.hash,
                    "model_hparams": hparams["model"],
                },
                model_save_path / f"{start_epoch + epoch}.pt",
            )


@torch.compile(mode="max-autotune", disable=False)
def _coarse_step(
    start_positions: torch.Tensor,
    heading_vectors: torch.Tensor,
    intensities: torch.Tensor,
    coarse_model: torch.nn.Module,
    coarse_optimizer: torch.optim.Optimizer,
    scaler_coarse: GradScaler,
    loss_fn: torch.nn.Module,
    hparams: dict,
):
    coarse_sample_ts, coarse_samples, coarse_sampling_distances = get_coarse_samples(
        start_positions,
        heading_vectors,
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
    fine_model: XRayModel,
    fine_optimizer: torch.optim.Optimizer,
    scaler_fine: GradScaler,
    loss_fn: torch.nn.Module,
    coarse_sample_ts: torch.Tensor,
    attenuation_coeff_pred: torch.Tensor,
    coarse_sampling_distances: torch.Tensor,
    hparams: dict,
):
    fine_samples, fine_sampling_distances = get_fine_samples(
        start_positions,
        heading_vectors,
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

    # samples = samples.to(intensities.device, dtype=intensities.dtype)
    # sampling_distances = sampling_distances.to(intensities.device, dtype=intensities.dtype)

    with autocast(device_type="cuda", enabled=hparams["training"]["use_amp"]):
        attenuation_coeff_pred = model(samples)
        attenuation_coeff_pred = attenuation_coeff_pred.reshape(
            hparams["training"]["batch_size"], -1
        )
        intensity_pred = log_beer_lambert_law(
            attenuation_coeff_pred, sampling_distances, **hparams["scaling"]
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
def _eval_fig(
    model: torch.nn.Module,
    size: tuple[int, int],
    device: torch.device,
    epoch: int,
    run: Run,
    tag: str,
):
    model.eval()

    yv = torch.linspace(-1, 1, size[0])
    zv = torch.linspace(-1, 1, size[1])
    yv, zv = torch.meshgrid(yv, zv, indexing="xy")
    coords = torch.stack([torch.zeros_like(yv), yv, zv], dim=-1)
    coords = coords.reshape(-1, 3)
    coords = coords.to(device)

    output = model(coords)
    output = output.reshape(size[1], size[0])

    fig = px.imshow(output.cpu().numpy(), color_continuous_scale="gray")
    run.track(Figure(fig), name="cross section", step=epoch, context={"model": tag})

    model.train()


if __name__ == "__main__":
    train()

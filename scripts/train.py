import torch
from torch import autocast, GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from ctnerf.models import XRayModel
from ctnerf.dataloading import XRayDataset
from ctnerf.utils import get_data_dir
from ctnerf.rays import get_coarse_samples, get_fine_samples, log_beer_lambert_law
from tqdm import tqdm
from aim import Run, Figure
import plotly.express as px
import datetime
from pathlib import Path


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.allow_tf32 = True


def train():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run = Run()
    run["hparams"] = {
        "name": "dev-testing",
        "n_layers": 12,
        "layer_dim": 256,
        "L": 20,
        "lr": 0.0001,
        "batch_size": 4096,
        "num_coarse_samples": 64,
        "num_fine_samples": 128,
        "device": "cuda:0",
        "s": 1,
        "k": 0.1,
        "slice_size_cm": 0.15234375 * 512,  # TODO: extract automatically?
        "dtype": "bfloat16",
        "use_amp": True,
    }

    if run["hparams"]["dtype"] == "bfloat16":
        dtype = torch.bfloat16
    elif run["hparams"]["dtype"] == "float16":
        dtype = torch.float16
    elif run["hparams"]["dtype"] == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {run['hparams']['dtype']}")

    model_save_path = Path(__file__).parents[1] / f"models/{run['hparams']['name']}/{timestamp}"
    model_save_path.mkdir(parents=True, exist_ok=True)

    datapath = get_data_dir() / "xrays" / "2 AC_CT_TBody"
    dataset = XRayDataset(datapath, s=run["hparams"]["s"], k=run["hparams"]["k"], dtype=dtype)
    dataloader = DataLoader(dataset, batch_size=run["hparams"]["batch_size"], shuffle=True)

    model_coarse = XRayModel(
        n_layers=run["hparams"]["n_layers"],
        layer_dim=run["hparams"]["layer_dim"],
        L=run["hparams"]["L"],
    )
    model_coarse.to(run["hparams"]["device"])
    model_fine = XRayModel(
        n_layers=run["hparams"]["n_layers"],
        layer_dim=run["hparams"]["layer_dim"],
        L=run["hparams"]["L"],
    )
    model_fine.to(run["hparams"]["device"])

    mse_loss = MSELoss(reduction="none")
    optimizer_coarse = Adam(model_coarse.parameters(), lr=run["hparams"]["lr"], fused=True)
    optimizer_fine = Adam(model_fine.parameters(), lr=run["hparams"]["lr"], fused=True)

    scaler_coarse = GradScaler()
    scaler_fine = GradScaler()

    total_batches = 0
    for epoch in range(1000):
        for start_positions, heading_vectors, intensities in tqdm(dataloader):
            # start_positions = start_positions.to(run["hparams"]["device"], dtype=dtype)
            # heading_vectors = heading_vectors.to(run["hparams"]["device"], dtype=dtype)
            intensities = intensities.to(run["hparams"]["device"], dtype=dtype)

            (
                coarse_sample_ts,
                coarse_attenuation_coeff_pred,
                coarse_sampling_distances,
                coarse_loss,
            ) = _coarse_step(
                start_positions,
                heading_vectors,
                intensities,
                model_coarse,
                optimizer_coarse,
                scaler_coarse,
                mse_loss,
                run["hparams"],
            )

            loss = _fine_step(
                start_positions,
                heading_vectors,
                intensities,
                model_fine,
                optimizer_fine,
                scaler_fine,
                mse_loss,
                coarse_sample_ts,
                coarse_attenuation_coeff_pred,
                coarse_sampling_distances,
                run["hparams"],
            )
            total_batches += 1
            run.track(loss.item(), name="loss", step=total_batches)
            run.track(coarse_loss.item(), name="coarse_loss", step=total_batches)

        # generate cross section
        _eval_fig(model_fine, run["hparams"]["device"], epoch, run, "fine")
        _eval_fig(model_coarse, run["hparams"]["device"], epoch, run, "coarse")

        if epoch % 1 == 0:
            torch.save(model_coarse.state_dict(), model_save_path / f"{epoch}_coarse.pt")
            torch.save(model_fine.state_dict(), model_save_path / f"{epoch}_fine.pt")


@torch.compile(mode="max-autotune", disable=False)
def _coarse_step(
    start_positions: torch.Tensor,
    heading_vectors: torch.Tensor,
    intensities: torch.Tensor,
    model_coarse: torch.nn.Module,
    optimizer_coarse: torch.optim.Optimizer,
    scaler_coarse: GradScaler,
    loss_fn: torch.nn.Module,
    hparams: dict,
):
    coarse_sample_ts, coarse_samples, coarse_sampling_distances = get_coarse_samples(
        start_positions,
        heading_vectors,
        hparams["num_coarse_samples"],
    )

    loss, attenuation_coeff_pred = _forward_backward(
        intensities,
        model_coarse,
        optimizer_coarse,
        scaler_coarse,
        loss_fn,
        coarse_samples,
        coarse_sampling_distances,
        hparams,
    )

    # attenuation_coeff_pred_ = attenuation_coeff_pred.detach().cpu()
    # coarse_sampling_distances_ = coarse_sampling_distances.detach().cpu()
    # loss_ = loss.detach().cpu()

    # del loss
    # del attenuation_coeff_pred
    # del coarse_samples
    # del coarse_sampling_distances

    return (
        coarse_sample_ts,
        attenuation_coeff_pred.detach().cpu(),
        coarse_sampling_distances,
        loss.detach().cpu(),
    )


@torch.compile(mode="max-autotune", disable=False)
def _fine_step(
    start_positions: torch.Tensor,
    heading_vectors: torch.Tensor,
    intensities: torch.Tensor,
    model_fine: XRayModel,
    optimizer_fine: torch.optim.Optimizer,
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
        hparams["num_fine_samples"],
    )

    loss, _ = _forward_backward(
        intensities,
        model_fine,
        optimizer_fine,
        scaler_fine,
        loss_fn,
        fine_samples,
        fine_sampling_distances,
        hparams,
    )

    # del _
    # del output
    # del fine_samples
    # del fine_sampling_distances

    return loss.detach().cpu()


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

    samples = samples.to(intensities.device, dtype=intensities.dtype)
    sampling_distances = sampling_distances.to(intensities.device, dtype=intensities.dtype)

    with autocast(device_type="cuda", enabled=hparams["use_amp"]):
        attenuation_coeff_pred = model(samples)
        attenuation_coeff_pred = attenuation_coeff_pred.reshape(hparams["batch_size"], -1)
        intensity_pred = log_beer_lambert_law(
            attenuation_coeff_pred,
            sampling_distances,
            hparams["s"],
            hparams["k"],
            hparams["slice_size_cm"],
        )
        loss = loss_fn(intensity_pred, intensities)
        loss = torch.sum(loss)

    if hparams["use_amp"]:
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
    device: torch.device,
    epoch: int,
    run: Run,
    tag: str,
):
    model.eval()
    yv = torch.linspace(-1, 1, 512)
    zv = torch.linspace(-1, 1, 536)
    yv, zv = torch.meshgrid(yv, zv, indexing="ij")
    coords = torch.stack([torch.zeros_like(yv), yv, zv], dim=-1)
    coords = coords.reshape(-1, 3)
    coords = coords.to(device)

    output = model(coords)
    output = output.reshape(512, 536).T

    fig = px.imshow(output.cpu().numpy(), color_continuous_scale="gray")
    run.track(Figure(fig), name="cross section", step=epoch, context={"model": tag})

    model.train()


if __name__ == "__main__":
    train()

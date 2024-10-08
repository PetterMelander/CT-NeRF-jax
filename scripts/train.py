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



torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.allow_tf32 = True


def train():

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run = Run()
    run["hparams"] = {
        "name": "dev-testing",
        "n_layers": 8,
        "layer_dim": 256,
        "L": 10,
        "lr": 0.0001,
        "batch_size": 4096,
        "num_coarse_samples" : 64,
        "num_fine_samples" : 128,
        "device": "cuda:0",
        "s": 1,
        "k": 0.1,
        "slice_size_cm": 0.15234375 * 512, # TODO: extract automatically?
    }
    model_save_path = Path(__file__).parents[1] / f"models/{run['hparams']['name']}/{timestamp}"
    model_save_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(run["hparams"]["device"])

    datapath = get_data_dir() / "xrays" / "2 AC_CT_TBody"
    dataset = XRayDataset(datapath, s=run["hparams"]["s"], k=run["hparams"]["k"])
    dataloader = DataLoader(dataset, batch_size=run["hparams"]["batch_size"], shuffle=True)

    model_coarse = XRayModel(
        n_layers=run["hparams"]["n_layers"],
        layer_dim=run["hparams"]["layer_dim"],
        L=run["hparams"]["L"],
    )
    model_coarse.to(device)
    model_fine = XRayModel(
        n_layers=run["hparams"]["n_layers"],
        layer_dim=run["hparams"]["layer_dim"],
        L=run["hparams"]["L"],
    )
    model_fine.to(device)

    mse_loss = MSELoss(reduction="none")
    optimizer_coarse = Adam(model_coarse.parameters(), lr=run["hparams"]["lr"], fused=True)
    optimizer_fine = Adam(model_fine.parameters(), lr=run["hparams"]["lr"], fused=True)

    scaler_coarse = GradScaler()
    scaler_fine = GradScaler()

    total_batches = 0
    for epoch in range(1000):
        for start_positions, heading_vectors, intensities in tqdm(dataloader):
            loss, coarse_loss = _train_step(
                start_positions,
                heading_vectors,
                intensities,
                run["hparams"]["slice_size_cm"],
                model_coarse,
                model_fine,
                optimizer_coarse,
                optimizer_fine,
                scaler_coarse,
                scaler_fine,
                mse_loss,
                run["hparams"]["num_coarse_samples"],
                run["hparams"]["num_fine_samples"],
                run["hparams"]["batch_size"],
                run["hparams"]["s"],
                run["hparams"]["k"],
                run["hparams"]["device"],
            )
            total_batches += 1
            run.track(loss.item(), name="loss", step=total_batches)
            run.track(coarse_loss.item(), name="coarse_loss", step=total_batches)
        
        # generate cross section
        _eval_fig(model_fine, device, epoch, run, "fine")
        _eval_fig(model_coarse, device, epoch, run, "coarse")


        if epoch % 1 == 0:
            torch.save(model_coarse.state_dict(), model_save_path / f"{epoch}_coarse.pt")
            torch.save(model_fine.state_dict(), model_save_path / f"{epoch}_fine.pt")


@torch.compile(mode="max-autotune", disable=False)
def _train_step(
        start_positions: torch.Tensor,
        heading_vectors: torch.Tensor,
        intensities: torch.Tensor,
        slice_size_cm: float,
        model_coarse: torch.nn.Module,
        model_fine: torch.nn.Module,
        optimizer_coarse: torch.optim.Optimizer,
        optimizer_fine: torch.optim.Optimizer,
        scaler_coarse: GradScaler,
        scaler_fine: GradScaler,
        loss_fn: torch.nn.Module,
        num_coarse_samples: int,
        num_fine_samples: int,
        batch_size: int,
        s: float,
        k: float,
        device: torch.device
        ) -> None:
    
    intensities = intensities.to(device, dtype=torch.bfloat16)

    coarse_sample_ts, attenuation_coeff_pred, coarse_sampling_distances, coarse_loss = _coarse_step(
        start_positions,
        heading_vectors,
        intensities,
        slice_size_cm,
        model_coarse,
        optimizer_coarse,
        scaler_coarse,
        loss_fn,
        num_coarse_samples,
        batch_size,
        s,
        k,
        device
    )

    loss = _fine_step(
        start_positions,
        heading_vectors,
        intensities,
        slice_size_cm,
        model_fine,
        optimizer_fine,
        scaler_fine,
        loss_fn,
        num_fine_samples,
        num_coarse_samples,
        batch_size,
        s,
        k,
        device,
        coarse_sample_ts,
        attenuation_coeff_pred,
        coarse_sampling_distances
    )

    del intensities

    return loss, coarse_loss


def _coarse_step(
        start_positions: torch.Tensor,
        heading_vectors: torch.Tensor,
        intensities: torch.Tensor,
        slice_size_cm: float,
        model_coarse: torch.nn.Module,
        optimizer_coarse: torch.optim.Optimizer,
        scaler_coarse: GradScaler,
        loss_fn: torch.nn.Module,
        num_coarse_samples: int,
        batch_size: int,
        s: float,
        k: float,
        device: torch.device
):
    model_coarse.train()

    coarse_sample_ts, coarse_samples, coarse_sampling_distances = get_coarse_samples(
        start_positions, 
        heading_vectors, 
        num_coarse_samples,
    )
    coarse_samples = coarse_samples.to(device, dtype=torch.bfloat16)
    coarse_sampling_distances = coarse_sampling_distances.to(device, dtype=torch.bfloat16)

    with autocast(device_type="cuda"):
        attenuation_coeff_pred = model_coarse(coarse_samples)
        attenuation_coeff_pred = attenuation_coeff_pred.reshape(batch_size, num_coarse_samples)
        intensity_pred = log_beer_lambert_law(attenuation_coeff_pred, coarse_sampling_distances, s, k, slice_size_cm)
        loss = loss_fn(intensity_pred, intensities)
        loss = torch.sum(loss)
    scaler_coarse.scale(loss).backward()
    scaler_coarse.step(optimizer_coarse)
    scaler_coarse.update()
    optimizer_coarse.zero_grad(set_to_none=True)

    attenuation_coeff_pred_ = attenuation_coeff_pred.detach().cpu()
    coarse_sampling_distances_ = coarse_sampling_distances.detach().cpu()
    loss_ = loss.detach().cpu()

    del loss
    del intensity_pred
    del attenuation_coeff_pred
    del coarse_samples
    del coarse_sampling_distances

    return coarse_sample_ts, attenuation_coeff_pred_, coarse_sampling_distances_, loss_

def _fine_step(
        start_positions: torch.Tensor,
        heading_vectors: torch.Tensor,
        intensities: torch.Tensor,
        slice_size_cm: float,
        model_fine: XRayModel,
        optimizer_fine: torch.optim.Optimizer,
        scaler_fine: GradScaler,
        loss_fn: torch.nn.Module,
        num_fine_samples: int,
        num_coarse_samples: int,
        batch_size: int,
        s: float,
        k: float,
        device: torch.device,
        coarse_sample_ts: torch.Tensor,
        attenuation_coeff_pred: torch.Tensor,
        coarse_sampling_distances: torch.Tensor,
):
    
    model_fine.train()
    fine_samples, fine_sampling_distances = get_fine_samples(
        start_positions,
        heading_vectors,
        coarse_sample_ts,
        attenuation_coeff_pred,
        coarse_sampling_distances,
        num_fine_samples
    )

    fine_samples = fine_samples.to(device, dtype=torch.bfloat16)
    fine_sampling_distances = fine_sampling_distances.to(device, dtype=torch.bfloat16)
    
    with autocast(device_type="cuda"):
        output = model_fine(fine_samples)
        output = output.reshape(batch_size, num_coarse_samples + num_fine_samples)
        output = log_beer_lambert_law(output, fine_sampling_distances, s, k, slice_size_cm)
        loss = loss_fn(output, intensities)
        loss = torch.sum(loss)
    scaler_fine.scale(loss).backward()
    scaler_fine.step(optimizer_fine)
    scaler_fine.update()
    optimizer_fine.zero_grad(set_to_none=True)

    loss = loss.detach().cpu()
    del output
    del fine_samples
    del fine_sampling_distances

    return loss


def _forward_backward(
    intensities: torch.Tensor,
    slice_size_cm: float,
    model: XRayModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: torch.nn.Module,
    batch_size: int,
    s: float,
    k: float,
    samples: torch.Tensor,
    sampling_distances: torch.Tensor,
):
    
    samples = samples.to(intensities.device, dtype=torch.bfloat16)
    sampling_distances = sampling_distances.to(intensities.device, dtype=torch.bfloat16)

    with autocast(device_type="cuda"):
        attenuation_coeff_pred = model(samples)
        attenuation_coeff_pred = attenuation_coeff_pred.reshape(batch_size, -1)
        intensity_pred = log_beer_lambert_law(attenuation_coeff_pred, sampling_distances, s, k, slice_size_cm)
        loss = loss_fn(intensity_pred, intensities)
        loss = torch.sum(loss)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
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

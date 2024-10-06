import torch
from ctnerf.models import XRayModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from ctnerf.dataloading import XRayDataset
from torch.nn import MSELoss
from aim import Run, Figure
import plotly.express as px
from ctnerf.utils import get_data_dir
from ctnerf.rays import get_samples, log_beer_lambert_law
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
        "num_samples" : 128,
        "device": "cuda:0",
        "s": 1,
        "k": 0.1,
        "slice_size_cm": 1.5234375 * 512 # TODO: extract automatically?
    }
    model_save_path = Path(__file__).parents[1] / f"models/{run['hparams']['name']}/{timestamp}"
    model_save_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(run["hparams"]["device"])

    datapath = get_data_dir() / "xrays" / "2 AC_CT_TBody"
    dataset = XRayDataset(datapath, s=run["hparams"]["s"], k=run["hparams"]["k"])
    dataloader = DataLoader(dataset, batch_size=run["hparams"]["batch_size"], shuffle=True)

    model = XRayModel(
        n_layers=run["hparams"]["n_layers"],
        layer_dim=run["hparams"]["layer_dim"],
        L=run["hparams"]["L"],
    )
    model.to(device)

    mse_loss = MSELoss(reduction="none")
    optimizer = Adam(model.parameters(), lr=run["hparams"]["lr"], fused=True)

    total_batches = 0
    for epoch in range(1000):
        for start_positions, heading_vectors, ray_bounds, intensities in tqdm(dataloader):
            # start_positions = start_positions.to(device)
            # heading_vectors = heading_vectors.to(device)
            # ray_bounds = ray_bounds.to(device)
            # intensities = intensities.to(device)
            loss = _train_step(
                start_positions,
                heading_vectors,
                ray_bounds,
                intensities,
                run["hparams"]["slice_size_cm"],
                model,
                optimizer,
                mse_loss,
                run["hparams"]["num_samples"],
                run["hparams"]["batch_size"],
                run["hparams"]["s"],
                run["hparams"]["k"],
                device
            )
            total_batches += 1
            run.track(loss.item(), name="loss", step=total_batches)
        
        # generate cross section
        _eval_fig(model, device, epoch, run)

        if epoch % 1 == 0:
            torch.save(model.state_dict(), model_save_path / f"{epoch}.pt")


@torch.compile(mode="max-autotune")
def _train_step(
        start_positions: torch.Tensor,
        heading_vectors: torch.Tensor,
        ray_bounds: torch.Tensor,
        intensities: torch.Tensor,
        slice_size_cm: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        num_samples: int,
        batch_size: int,
        s: float,
        k: float,
        device: torch.device
        ) -> None:

    sampled_points, sampling_distances = get_samples(
        start_positions, 
        heading_vectors, 
        ray_bounds, 
        num_samples,
        slice_size_cm
    )
    sampled_points = sampled_points.to(device)
    sampling_distances = sampling_distances.to(device)
    intensities = intensities.to(device)

    optimizer.zero_grad(set_to_none=True)
    output = model(sampled_points)
    output = output.reshape(batch_size, num_samples)
    output = log_beer_lambert_law(output, sampling_distances, s=s, k=k)
    loss = loss_fn(output, intensities)
    loss = torch.sum(loss)

    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def _eval_fig(
    model: torch.nn.Module,
    device: torch.device,
    epoch: int,
    run: Run
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
    run.track(Figure(fig), name="cross_section", step=epoch)

    model.train()



if __name__ == "__main__":
    train()

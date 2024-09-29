import torch
from devtools.models import XRayModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from dataloading import XRayDataset
from torch.nn import MSELoss
from aim import Run, Figure
import plotly.express as px
from devtools.utils import get_data_dir
from devtools.rays import get_samples, log_beer_lambert_law
import math



def train():

    torch.autograd.set_detect_anomaly(True)

    run = Run()
    run["hparams"] = {
        "n_layers": 8,
        "layer_dim": 256,
        "L": 10,
        "lr": 0.001,
        "batch_size": 4096,
        "num_samples" : 128,
        "proportional_sampling": False,
        "device": "cuda:0",
        "s": math.log(2),
        "k": 1
    }
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

    optimizer = Adam(model.parameters(), lr=run["hparams"]["lr"])

    total_batches = 0
    for epoch in range(100):
        for i, (start_positions, heading_vectors, ray_bounds, intensities) in enumerate(tqdm(dataloader)):
            start_positions = start_positions.to(device)
            heading_vectors = heading_vectors.to(device)
            ray_bounds = ray_bounds.to(device)
            intensities = intensities.to(device)

            sampled_points, sampling_distances = get_samples(
                start_positions, 
                heading_vectors, 
                ray_bounds, 
                run["hparams"]["num_samples"], 
                run["hparams"]["proportional_sampling"]
            )

            optimizer.zero_grad()
            output = model(sampled_points)
            output = output.reshape(run["hparams"]["batch_size"], run["hparams"]["num_samples"])
            # output = torch.nan_to_num(output)
            output = log_beer_lambert_law(output, sampling_distances, s=run["hparams"]["s"], k=run["hparams"]["k"])
            loss = mse_loss(output, intensities)
            loss = torch.sum(loss)

            loss.backward()
            optimizer.step()
            total_batches += 1
            run.track(loss.item(), name="loss", step=total_batches)
        
            if i % 100 == 0:
                # generate cross section
                model.eval()
                with torch.no_grad():
                    yv = torch.linspace(-1, 1, 512)
                    zv = torch.linspace(-1, 1, 536)
                    yv, zv = torch.meshgrid(yv, zv, indexing="ij")
                    yv = yv.to(device)
                    zv = zv.to(device)
                    coords = torch.stack([torch.zeros_like(yv), yv, zv], dim=-1)
                    coords = coords.reshape(-1, 3)
                    coords = coords.to(device)

                    output = model(coords)
                    output = output.reshape(512, 536)

                    fig = px.imshow(output.cpu().numpy())
                    run.track(Figure(fig), name="cross_section", step=int(total_batches/100))

                model.train()
                        




if __name__ == "__main__":
    train()

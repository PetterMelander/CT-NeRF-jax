from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from ctnerf.models import XRayModel


@torch.no_grad()
def generate_ct(
    model_path: Path,
    n_layers: int,
    layer_size: int,
    pos_embed_dim: int,
    ct_path: Path,
    img_size: tuple[int, int, int],
    chunk_size: int,
    device: torch.device,
) -> None:
    model = XRayModel(n_layers, layer_size, pos_embed_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    x = torch.arange(-1, 1, 2 / img_size[0])
    y = torch.arange(-1, 1, 2 / img_size[1])
    z = torch.arange(-1, 1, 2 / img_size[2])
    coords = torch.stack(torch.meshgrid((x, y, z), indexing="ij"), dim=-1)
    coords = coords.view(-1, 3)

    coords = coords.to(device)

    # To avoid oom, inference is done in batches and result stored on cpu
    output = torch.tensor([], device="cpu")
    coords = coords.split(chunk_size, dim=0)
    for chunk in tqdm(coords, desc="Generating", total=len(coords)):
        chunk = chunk.to(device)
        output_chunk = model(chunk)
        output_chunk = output_chunk.view(-1)
        output = torch.cat((output, output_chunk.cpu()))

    # Convert to hounsfield
    mu_air = 0.0002504  # 50 keV, per cm (https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html)
    mu_water = 0.2269  # 50 keV, per cm (https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)
    output = 1000 * (output - mu_water) / (mu_water - mu_air)

    output = output.reshape(img_size[0], img_size[1], img_size[2])

    # TODO: handle voxel sizes
    img = nib.nifti1.Nifti1Image(output.numpy, np.eye(4))
    nib.nifti1.save(img, ct_path)

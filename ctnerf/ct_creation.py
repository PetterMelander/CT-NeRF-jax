from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from ctnerf.models import XRayModel


@torch.no_grad()
def generate_ct(
    model_path: Path,
    ct_path: Path,
    img_size: tuple[int, int, int],
    chunk_size: int,
    metadata,
    device: torch.device,
) -> None:
    
    saved_checkpoint = torch.load(model_path, weights_only=True)

    model = XRayModel(**saved_checkpoint["model_hparams"])
    model.load_state_dict(saved_checkpoint["fine_model"])
    model.to(device)
    model.eval()

    x = torch.linspace(-1, 1, img_size[0])
    y = torch.linspace(-1, 1, img_size[1])
    z = torch.linspace(-1, 1, img_size[2])

    coords = torch.stack(torch.meshgrid((x, y, z), indexing="xy"), dim=-1) # TODO: z, y, x?
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
    output = torch.permute(output, (2,1,0))

    ct_array = output.numpy().astype(np.int16)
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetDirection(metadata["direction"])
    ct_image.SetOrigin(metadata["origin"])

    for key, value in metadata["ct_meta"].items():
        ct_image.SetMetaData(key, value)

    original_spacing = np.array(metadata["spacing"])
    original_size = np.array(metadata["size"])
    new_size = np.array(img_size)
    new_spacing = original_spacing * original_size / new_size
    ct_image.SetSpacing(new_spacing)

    sitk.WriteImage(ct_image, ct_path)

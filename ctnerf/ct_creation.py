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
    metadata: dict,
    img_size: tuple[int, int, int] | None = None,
    spacing: tuple[float, float, float] | None = None,
    origin: tuple[float, float, float] | None = None,
    direction: tuple[float] | None = None,
    chunk_size: int = 4096 * 64,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Generates a CT image from a trained model and saves it to the specified path. Either img_size
    or spacing must be specified. If both are specified, spacing takes precedence.

    Args:
        model_path (Path): Path to the model checkpoint file.
        ct_path (Path): Destination path to save the generated CT image.
        metadata (dict): Metadata information needed for setting the CT image properties.
        img_size (list[int, int, int]): The size (dimensions) of the output CT image.
            If None, spacing is used to determine the size.
        spacing (list[float, float, float]): The spacing (spacing between voxels) of the output CT image.
            If None, img_size is used to determine the spacing.
        origin (list[float, float, float]): The origin of the output CT image in sitk notation.
            If None, (0, 0, 0) is used.
        direction (list[float, float, float]): The direction of the output CT image in sitk notation.
            If None, (1, 0, 0, 0, 1, 0, 0, 0, 1) is used.
        chunk_size (int): Number of coordinate points to process in a single batch to avoid OOM errors.
        device (torch.device): The device to run the model inference on.

    Returns:
        None
    """

    # Define image size and spacing, giving x the same value as y
    original_size = [metadata["size"][0]] + metadata["size"]
    original_spacing = [metadata["spacing"][0]] + metadata["spacing"]

    if origin is None:
        origin = [0, 0, 0]

    if direction is None:
        direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    if img_size is None and spacing is None:
        raise ValueError("Either img_size or spacing must be specified.")

    if spacing is not None:
        original_spacing = np.array(original_spacing)
        original_size = np.array(original_size)
        new_spacing = np.array(spacing)
        img_size = (original_size / original_spacing * new_spacing).astype(int)
        
    elif img_size is not None:
        original_spacing = np.array(original_spacing)
        original_size = np.array(original_size)
        new_size = np.array(img_size)
        spacing = original_spacing * original_size / new_size

    saved_checkpoint = torch.load(model_path, weights_only=True)

    model = XRayModel(**saved_checkpoint["model_hparams"])
    model.load_state_dict(saved_checkpoint["coarse_model"])
    model.to(device)
    model.eval()

    x = torch.linspace(-1, 1, img_size[0])
    y = torch.linspace(-1, 1, img_size[1])
    z = torch.linspace(-1, 1, img_size[2])

    coords = torch.stack(torch.meshgrid((x, y, z), indexing="xy"), dim=-1)
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
    output = torch.permute(output, (2, 1, 0))
    output = output.clamp_min(-1024)

    ct_array = output.numpy().astype(np.int16)
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetDirection(direction)
    ct_image.SetOrigin(origin)
    ct_image.SetSpacing(spacing)

    if "ct_meta" in metadata:
        for key, value in metadata["ct_meta"].items():
            ct_image.SetMetaData(key, value)

    sitk.WriteImage(ct_image, ct_path)

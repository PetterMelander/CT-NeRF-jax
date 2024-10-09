import torch
from pathlib import Path
import numpy as np
import nrrd
from monai.data import MetaTensor
from monai.transforms.spatial.functional import rotate
from PIL import Image



def generate_xrays(
    ct_path: Path,
    output_dir: Path,
    angle_interval_size: int,
    max_angle: int,
    device: str
) -> None:

    output_dir.mkdir(exist_ok=True, parents=True)
    img = _read_nrrd(path=ct_path, device=device).detach()

    angles = ""
    for angle in range(0, max_angle, angle_interval_size):
        xray = _ct_to_xray(img, pixel_spacing=img.meta["spacing"][1,1]/10, angle=np.radians(angle))
        xray = Image.fromarray((xray * (2**16 - 1)).astype('uint16').squeeze(0))
        xray.save(output_dir / f"{angle}.png")
        angles += f"{angle}\n"

    with (open(output_dir / "angles.txt", "w")) as f:
        f.write(angles)


def _read_nrrd(path: Path, device: str = "cpu") -> torch.Tensor:
    """
    Reads a .nrrd file and returns a torch.Tensor with shape (C, H, W, D).
    Makes some assumptions about the orientation of the image.

    Args:
        path (Path): path to .nrrd file
        device (str): device to store the image on

    Returns:
        torch.Tensor: CT image with shape (C, H, W, D)
    """
    img, header = nrrd.read(path, index_order='C')
    img = np.flip(img, axis=0).copy()
    header["spacing"] = np.fliplr(np.flipud(header["space directions"]))
    header["origin"] = np.flip(header["space origin"])
    header["size"] = np.flip(header["sizes"])
    img = MetaTensor(img, device=device, meta=header)
    img = img.unsqueeze(0).permute(0, 1, 3, 2)
    return img


def _ct_to_xray(
    img: torch.Tensor,
    angle: float = 0,
    pixel_spacing: float = 0.97
    ) -> torch.Tensor:
    """
    Turns a CT image into an X-ray image by using a discretized version of Beer-Lambert's law
    depth-wise, using CT values as attenuation coefficient.

    Args:
        img (torch.Tensor): CT image with shape (C, H, W, D)
        angle (float): angle that the X-ray was taken with. 0 degrees is frontal view
        pixel_spacing (float): pixel spacing in cm

    Returns:
        torch.Tensor: X-ray image with shape (C, H, W)
    """
    img = _hounsfield_to_attenuation(img)
    img = rotate(
        img, 
        angle=(angle, 0, 0),
        mode="bilinear", 
        padding_mode="zeros", 
        output_shape=img.shape[1:],
        align_corners=False,
        dtype=img.dtype,
        lazy=False,
        transform_info=False
        )
    img = torch.exp(-img.sum(axis=3) * pixel_spacing) # Beer-Lambert's law
    img = img.clamp(0, 1) # transmittance must be between 0 and 1
    return img
    

def _hounsfield_to_attenuation(self, img: torch.Tensor) -> torch.Tensor:
    """
    Reverses the formula for Hounsfield units to turn the CT image into a map of attenuation coefficients
    """
    mu_air = 0.0002504 # 50 keV, per cm (https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html)
    mu_water = 0.2269 # 50 keV, per cm (https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)
    img = img * (mu_water - mu_air) / 1000 + mu_water
    img[img < 0] = 0 # remove negative attenuation coefficients caused by padding with -1024
    return img

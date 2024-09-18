import monai.transforms
import torch
import monai
from pathlib import Path
import numpy as np
import nrrd
from monai.data import MetaTensor
from monai.transforms.spatial.functional import rotate

class NrrdReader(monai.transforms.Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, path: Path, device: str = "cpu") -> torch.Tensor:
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


class CtToXray(monai.transforms.Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img: torch.Tensor, angle: float = 0, pixel_spacing: float = 0.97) -> torch.Tensor:
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
        img = self.hounsfield_to_attenuation(img)
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
        img = -img # get the negative because that's how X-ray images are displayed
        return img
    

    def hounsfield_to_attenuation(self, img: torch.Tensor) -> torch.Tensor:
        """
        Reverses the formula for Hounsfield units to turn the CT image into a map of attenuation coefficients
        """
        mu_air = 0.0002504 # 50 keV, per cm (https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html)
        mu_water = 0.2269 # 50 keV, per cm (https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)
        return img * (mu_water - mu_air) / 1000 + mu_water


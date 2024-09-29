from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import math
from tqdm import tqdm
from devtools.rays import get_rays
import numpy as np



class XRayDataset(Dataset):


    @torch.no_grad()
    def __init__(
            self,
            xray_dir: Path,
            s: float = 1,
            k: float = 0,
            *args,
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)

        images = self._read_images(xray_dir)
        pixels_per_image = images[0].size[0] * images[0].size[1]
        self.len = len(images) * pixels_per_image

        # setup angles
        angles = []
        with open(xray_dir / "angles.txt", "r") as f:
            for line in f:
                angles += [math.radians(float(line.strip()))] * pixels_per_image
        angles = torch.tensor(angles)

        # setup positions
        positions = torch.zeros((self.len, 2))
        x = torch.linspace(0, images[0].size[0] - 1, images[0].size[0])
        y = torch.linspace(0, images[0].size[1] - 1, images[0].size[1])
        x, y = torch.meshgrid(x, y)
        positions = torch.stack((x, y), dim=-1).reshape(-1, 2).repeat(len(images), 1)

        # setup correpsponding intensities
        index = 0
        self.intensities = torch.zeros(self.len)
        for image in images:
            intensities = torch.tensor(np.array(image)).reshape(-1)
            self.intensities[index: index + pixels_per_image] = intensities
            index += pixels_per_image
        
        image_size = torch.tensor(images[0].size).expand(self.len, 2)
        self.start_positions, self.heading_vectors, self.ray_bounds = get_rays(positions, angles, image_size)
        
        # transform intensities to have values that are more evenly distributed
        self.intensities = self.intensities / (2**16 - 1) # TODO: find bit depth more programatically
        self.intensities = torch.log(self.intensities + k) / s
        self.intensities = torch.nan_to_num(self.intensities) # intensity 0 gives -inf after log


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.start_positions[index], self.heading_vectors[index], self.ray_bounds[index], self.intensities[index]
    

    def __len__(self) -> int:
        return self.len
    

    def _read_images(self, path: Path) -> list[Image.Image]:
        images = []
        for image_path in path.iterdir():
            if image_path.suffix == ".png":
                images.append(Image.open(image_path))
        return images

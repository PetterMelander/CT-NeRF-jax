import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ctnerf.rays import get_rays


class XRayDataset(Dataset):
    @torch.no_grad()
    def __init__(
        self,
        xray_dir: Path,
        s: float = 1,
        k: float = 0,
        dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        images, angles, self.len = self._read_images(xray_dir)

        # setup positions
        x = torch.linspace(0, images[0].size[0] - 1, images[0].size[0]) # TODO: extract img size from metadata
        y = torch.linspace(0, images[0].size[1] - 1, images[0].size[1])
        x, y = torch.meshgrid(x, y, indexing="ij")
        positions = torch.stack((x, y), dim=-1).reshape(-1, 2).repeat(len(images), 1)

        # setup corresponding intensities
        index = 0
        self.intensities = torch.zeros(self.len)
        for image in images:
            pixel_count = image.size[0] * image.size[1] # TODO: find constant img size from metadata
            intensities = torch.tensor(np.array(image, dtype=np.uint16)).T.reshape(-1) # why is this transposed? # TODO: find dtype from metadata
            self.intensities[index : index + pixel_count] = intensities
            index += pixel_count

        # transform intensities to have values that are more evenly distributed
        # TODO: find bit depth from metadata
        self.intensities = self.intensities / (2**16 - 1)
        self.intensities = torch.log(self.intensities + k) / s
        self.intensities = torch.nan_to_num(self.intensities)  # intensity 0 gives -inf after log

        image_size = torch.tensor(images[0].size) # TODO: get img size from metatdata
        self.start_positions, self.heading_vectors = get_rays(positions, angles, image_size)

        self.start_positions = self.start_positions.to(dtype=dtype)
        self.heading_vectors = self.heading_vectors.to(dtype=dtype)
        self.intensities = self.intensities.to(dtype=dtype)

        self.start_positions.requires_grad_(False)
        self.heading_vectors.requires_grad_(False)
        self.intensities.requires_grad_(False)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.start_positions[index], self.heading_vectors[index], self.intensities[index]

    def __len__(self) -> int:
        return self.len

    def _read_images(self, train_dir: Path) -> tuple[list[Image.Image], torch.Tensor, int]:
        with open(train_dir / "meta.json", "r") as f:
            metadata = json.load(f)

        images = []
        angles = []
        total_pixels = 0 # TODO: find constant img size from metadata
        for file, angle in metadata["file_angle_map"].items():
            image = Image.open(train_dir / file)
            pixel_count = image.size[0] * image.size[1]
            total_pixels += pixel_count
            images.append(image)
            angles += [(math.radians(float(angle)))] * pixel_count
        angles = torch.tensor(angles)

        return images, angles, total_pixels

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ctnerf.rays import get_rays
from ctnerf.utils import get_dataset_metadata


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
        """
        Initializes the XRayDataset by loading X-ray images and processing them to obtain
        angles, intensities, and pixel indices. Additionally, computes starting positions
        and heading vectors for ray tracing.

        Args:
            xray_dir (Path): Directory containing X-ray image files and metadata.
            s (float, optional): Scaling factor for intensity values. Defaults to 1.
            k (float, optional): Offset added to intensity values before applying log. Defaults to 0.
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to torch.float32.
            *args: Additional positional arguments passed to the base class.
            **kwargs: Additional keyword arguments passed to the base class.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)

        # Read metadata
        metadata = get_dataset_metadata(xray_dir)
        xray_size = metadata["size"]
        num_xrays = len(metadata["file_angle_map"])
        self.len = np.prod(xray_size) * num_xrays

        # Get angles, intensities and pixel indices from images
        angles, intensities, pixel_indices = self._read_images(xray_dir, metadata)

        # Scale intensities
        intensities = torch.log(intensities + k) / s
        self.intensities = torch.nan_to_num(intensities)  # intensity 0 gives -inf after log

        # Get positions and heading vectors in model space
        size_tensor = torch.tensor(xray_size)
        self.start_positions, self.heading_vectors = get_rays(pixel_indices, angles, size_tensor)

        # Tensor setup
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

    def _read_images(
        self, train_dir: Path, metadata: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reads xray images and returns angles, intensities and pixel indices.

        Args:
            train_dir (Path): The directory containing the xray images.
            metadata (dict): The metadata from the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: angles, intensities and pixel indices.
        """
        angles = []
        intensities = np.ndarray(0, dtype=np.float64)
        pixel_indices = torch.zeros(size=(0, 2))

        for file, angle in tqdm(metadata["file_angle_map"].items(), "Loading dataset"):
            # Read image
            xray_image = sitk.ReadImage(train_dir / file)
            xray_size = xray_image.GetSize()

            # Save angle for each pixel
            angles += [(np.radians(angle))] * np.prod(xray_size)

            # Save intensity for each pixel
            xray_intensities = sitk.GetArrayFromImage(xray_image)
            intensities = np.append(intensities, xray_intensities)

            # Save pixel indices corresponding to intensities and angles
            y = torch.linspace(0, xray_size[0] - 1, xray_size[0])
            z = torch.linspace(0, xray_size[1] - 1, xray_size[1])
            y, z = torch.meshgrid(y, z, indexing="xy")
            pixel_indices = torch.cat(
                [pixel_indices, torch.stack((y, z), dim=-1).reshape(-1, 2)], dim=0
            )

        angles = torch.tensor(angles)
        intensities = torch.tensor(intensities)

        return angles, intensities, pixel_indices

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import math
from tqdm import tqdm



class XRayDataset(Dataset):


    def __init__(self, xray_dir: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        images = self._read_images(xray_dir)
        pixels_per_image = images[0].size[0] * images[0].size[1]
        self.len = len(images) * pixels_per_image

        # setup angles
        angles = []
        with open(xray_dir / "angles.txt", "r") as f:
            for line in f:
                angles += [math.radians(float(line.strip()))] * pixels_per_image
        self.angles = torch.tensor(angles)

        # setup positions and intensities
        self.positions = torch.zeros((self.len, 2))
        self.intensities = torch.zeros(self.len)
        index = 0
        for image in tqdm(images, desc="Setting up positions and intensities"):
            for x in range(image.size[0]):
                for y in range(image.size[1]):
                    self.positions[index, 0] = x
                    self.positions[index, 1] = y
                    self.intensities[index] = image.getpixel((x, y))
                    index += 1
        
        # transform intensities to have values that are more evenly distributed
        self.intensities = self.intensities / (2**16 - 1) # TODO: find bit depth more programatically
        self.intensities = torch.log(self.intensities) # TODO: are these values reasonable or do they need to be scaled?


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.positions[index], self.angles[index], self.intensities[index]
    

    def __len__(self) -> int:
        return self.len
    

    def _read_images(self, path: Path) -> list[Image.Image]:
        images = []
        for image_path in path.iterdir():
            if image_path.suffix == ".png":
                images.append(Image.open(image_path))
        return images

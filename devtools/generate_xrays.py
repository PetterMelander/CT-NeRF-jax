from transforms.ct_to_xray import CtToXray, NrrdReader
from devtools.utils import get_data_dir
import torch
import numpy as np



datapath = get_data_dir()
ct_path = datapath / "ct_images" / "nrrd" / "2 AC_CT_TBody.nrrd"
output_dir = datapath / "xrays" / ct_path.stem
output_dir.mkdir(exist_ok=True, parents=True)
reader = NrrdReader()
xray = CtToXray()
img = reader(path=ct_path, device="cuda:0")

for angle in np.arange(0, 2*np.pi, 2*np.pi/32):
    xray_img = xray(img, pixel_spacing=img.meta["spacing"][1,1], angle=angle)
    torch.save(xray_img, output_dir / f"{angle}.pt")
    print(f"Saved {output_dir / f'{angle}.pt'}")

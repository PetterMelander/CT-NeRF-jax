from transforms.ct_to_xray import CtToXray, NrrdReader
from devtools.utils import get_data_dir
import math
from PIL import Image



datapath = get_data_dir()
ct_path = datapath / "ct_images" / "nrrd" / "2 AC_CT_TBody.nrrd"
output_dir = datapath / "xrays" / ct_path.stem
output_dir.mkdir(exist_ok=True, parents=True)
reader = NrrdReader()
xray = CtToXray()
img = reader(path=ct_path, device="cuda:0").detach()

angles = ""
for angle in range(0, 180, 5):
    xray_img = xray(img, pixel_spacing=img.meta["spacing"][1,1]/10, angle=math.radians(angle))
    xray_img = Image.fromarray((xray_img * (2**16 - 1)).astype('uint16').squeeze(0))
    xray_img.save(output_dir / f"{angle}.png")
    angles += f"{angle}\n"

with (open(output_dir / "angles.txt", "w")) as f:
    f.write(angles)

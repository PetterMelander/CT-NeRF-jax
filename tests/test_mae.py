# %%

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from ctnerf.image_creation.ct_creation import run_inference, tensor_to_sitk
from ctnerf.setup.config import get_inference_config
from ctnerf.utils import get_config_dir, get_ct_dir

# %%

conf = get_inference_config(get_config_dir() / "inference_config.yaml")
model = conf.coarse_model

generated_ct = run_inference(
    model,
    (512, 512, 536),
    4096 * 128,
    conf.attenuation_scaling_factor,
    conf.device,
)

# %%

ct_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
generated_ct = tensor_to_sitk(generated_ct, direction=ct_direction)
generated_ct = sitk.GetArrayFromImage(generated_ct)

source_ct_image = sitk.ReadImage(get_ct_dir() / "nrrd" / "2 AC_CT_TBody.nrrd")
source_ct_image = sitk.GetArrayFromImage(source_ct_image)

# %%

# generated_ct_t = np.transpose(generated_ct, (0, 2, 1))
generated_ct_t = generated_ct

mae = np.mean(np.abs(generated_ct_t - source_ct_image))
print(mae)

fig1 = plt.figure()
plt.imshow(generated_ct_t[:, :, 256], cmap="gray")

fig2 = plt.figure()
plt.imshow(source_ct_image[:, :, 256], cmap="gray")

plt.show()

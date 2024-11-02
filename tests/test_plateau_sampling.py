import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def pdf(x, ratio):
    erf1 = np.vectorize(math.erf)((x + ratio / 2) / (math.sqrt(2)))
    erf2 = np.vectorize(math.erf)((x - ratio / 2) / (math.sqrt(2)))
    return 1 / (4 * ratio / 2) * (erf1 - erf2)


ratio = 10
x = np.linspace(-10, 10, 10000)
plt.figure()
plt.plot(x, pdf(x, ratio))

x1 = torch.rand(100000).numpy() * ratio - (ratio / 2)
x2 = torch.randn(100000).numpy()

x = x1 + x2

# scale x to [t_min, t_max]
# t_min = -1
# t_max = 1
# x_min = np.min(x)
# x_max = np.max(x)
# x = (x - x_min) / (x_max - x_min) * (t_max - t_min) + t_min

plt.hist(x, bins=100, density=True)
plt.show()

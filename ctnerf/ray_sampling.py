"""Functions for sampling along rays."""

import torch


@torch.no_grad()
def uniform_sampling(n_samples: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample n_samples points evenly along the ray.

    Args:
        n_samples (int): number of samples
        batch_size (int): batch size
        device (torch.device): device
    Returns:
        torch.Tensor: shape (B, n_samples). Contains the sampled t's

    """
    interval_size = 2 / n_samples

    uniform_samples = torch.arange(0, n_samples, device=device).repeat(
        batch_size,
        1,
    )

    # Rescale each row to [t_min, t_max)
    uniform_samples = uniform_samples * interval_size

    perturbation = (
        torch.rand(batch_size, n_samples, device=device) * interval_size
    )

    return uniform_samples + perturbation


@torch.no_grad()
def cylinder_sampling(ray_bounds: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Sample n_samples points evenly along the ray inside the central cylinder.

    Args:
        ray_bounds (torch.Tensor): shape (B, 2). Ray bounds.
        n_samples (int): number of samples
    Returns:
        torch.Tensor: shape (B, n_samples). Contains the sampled t's

    """
    interval_size = ((ray_bounds[:, 1] - ray_bounds[:, 0]) / n_samples).unsqueeze(1)

    uniform_samples = torch.arange(0, n_samples, device=ray_bounds.device).repeat(
        ray_bounds.shape[0],
        1,
    )

    # Rescale each row to [t_min, t_max)
    uniform_samples = uniform_samples * interval_size + ray_bounds[:, 0].unsqueeze(1)

    perturbation = (
        torch.rand(ray_bounds.shape[0], n_samples, device=ray_bounds.device) * interval_size
    )

    return uniform_samples + perturbation


@torch.no_grad()
def plateau_sampling(
    n_samples: int,
    plateau_ratio: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample n_samples along the ray using a plateau distribution beginning at 0.

    Args:
        n_samples (int): number of samples
        plateau_ratio (float): ratio of plateau width to normal distribution
        batch_size (int): batch size
        device (torch.device): device
    Returns:
        torch.Tensor: shape (B, n_samples). Contains the sampled t's

    """
    x1 = torch.rand(batch_size, n_samples, device=device) * plateau_ratio - (plateau_ratio / 2)
    x2 = torch.randn(batch_size, n_samples, device=device)
    samples = x1 + x2
    samples = torch.sort(samples, dim=1)[0]

    # Rescale each row to [0, 2)
    s_min = samples[:, 0].unsqueeze(1)
    s_max = samples[:, -1].unsqueeze(1)
    return (samples - s_min) / (s_max - s_min) * 2


@torch.no_grad()
def plateau_cylinder_sampling(
    ray_bounds: torch.Tensor,
    n_samples: int,
    plateau_ratio: float,
) -> torch.Tensor:
    """Sample n_samples points along the ray using plateau sampling within the cylinder bounds.

    Args:
        ray_bounds (torch.Tensor): shape (B, 2). Ray bounds.
        n_samples (int): number of samples
        plateau_ratio (float): ratio of plateau width to standard deviation
    Returns:
        torch.Tensor: shape (B, n_samples). Contains the sampled t's

    """
    x1 = torch.rand(ray_bounds.shape[0], n_samples, device=ray_bounds.device) * plateau_ratio - (
        plateau_ratio / 2
    )
    x2 = torch.randn(ray_bounds.shape[0], n_samples, device=ray_bounds.device)
    samples = x1 + x2
    samples = torch.sort(samples, dim=1)[0]

    # Rescale each row to [t_min, t_max)
    t_min = ray_bounds[:, 0].unsqueeze(1)
    t_max = ray_bounds[:, 1].unsqueeze(1)
    s_min = samples[:, 0].unsqueeze(1)
    s_max = samples[:, -1].unsqueeze(1)
    return (samples - s_min) / (s_max - s_min) * (t_max - t_min) + t_min


@torch.no_grad()
def fine_sampling(
    num_samples: int,
    coarse_sample_values: torch.Tensor,
    coarse_sampling_distances: torch.Tensor,
) -> torch.Tensor:
    """Sample n_samples points along the ray using the density based sampling from the NeRF paper.

    Args:
        num_samples (int): number of samples
        coarse_sample_values (torch.Tensor): shape (B, n_samples). Contains the output values of the
            coarse model.
        coarse_sampling_distances (torch.Tensor): shape (B, n_samples). Contains the sampling
            distances of the coarse sampling
    Returns:
        torch.Tensor: shape (B, n_samples). Contains the sampled t's

    """
    # compute a "cdf" of the intensity found by the coarse model, accounting for sampling distances
    pdf = coarse_sample_values * coarse_sampling_distances
    pdf = pdf / torch.sum(pdf, dim=1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=1)
    cdf[..., -1] = 1 + 1e-5  # to avoid rounding causing index out of bounds if x is close to 1

    # inverse transform sampling
    # each random x will fall between two sampling distances, lower and upper
    x = torch.rand(coarse_sample_values.shape[0], num_samples, device=coarse_sample_values.device)
    inds = torch.searchsorted(cdf, x, right=False)
    cum_sampling_distances = torch.cumsum(coarse_sampling_distances, dim=1)
    cum_sampling_distances = torch.cat(
        [
            torch.zeros((cum_sampling_distances.shape[0], 1), device=cum_sampling_distances.device),
            cum_sampling_distances,
        ],
        dim=1,
    )
    lower = torch.gather(cum_sampling_distances, 1, inds)
    upper = torch.gather(cum_sampling_distances, 1, inds + 1)

    # uniformly sample between lower and upper
    t = torch.rand(coarse_sample_values.shape[0], num_samples, device=coarse_sample_values.device)
    return lower + t * (upper - lower)


@torch.no_grad()
def edge_focused_fine_sampling(
    num_samples: int,
    coarse_sample_values: torch.Tensor,
    coarse_sampling_distances: torch.Tensor,
) -> torch.Tensor:
    """Edge focused fine sampling.

    Sample n_samples points along the ray using a sampling strategy based on extra sampling
    around edges found by the coarse model.

    Args:
        num_samples (int): number of samples
        coarse_sample_values (torch.Tensor): shape (B, n_samples). Contains the output values of the
            coarse model.
        coarse_sampling_distances (torch.Tensor): shape (B, n_samples). Contains the sampling
            distances of the coarse model.

    Returns:
        torch.Tensor: shape (B, n_samples). Contains the sampled t's

    """
    zeros = torch.zeros([coarse_sample_values.shape[0], 1], device=coarse_sample_values.device)
    diff = torch.abs(torch.diff(coarse_sample_values, dim=1, append=zeros))

    # compute a "cdf" of the intensity found by the coarse model, accounting for sampling distances
    pdf = diff / coarse_sampling_distances
    pdf = pdf / torch.sum(pdf, dim=1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=1)
    cdf[..., -1] = 1 + 1e-5  # to avoid rounding causing index out of bounds if x is close to 1

    # inverse transform sampling
    # each random x will fall between two sampling distances, lower and upper
    x = torch.rand(coarse_sample_values.shape[0], num_samples, device=coarse_sample_values.device)
    inds = torch.searchsorted(cdf, x, right=False)
    cum_sampling_distances = torch.cumsum(coarse_sampling_distances, dim=1)
    cum_sampling_distances = torch.cat(
        [
            torch.zeros((cum_sampling_distances.shape[0], 1), device=cum_sampling_distances.device),
            cum_sampling_distances,
        ],
        dim=1,
    )
    lower = torch.gather(cum_sampling_distances, 1, inds)
    upper = torch.gather(cum_sampling_distances, 1, inds + 1)

    # uniformly sample between lower and upper
    t = torch.rand(coarse_sample_values.shape[0], num_samples, device=coarse_sample_values.device)
    return lower + t * (upper - lower)

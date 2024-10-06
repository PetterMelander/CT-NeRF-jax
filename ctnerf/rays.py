import torch



def log_beer_lambert_law(
        attenuation_coeffs: torch.Tensor, 
        distances: torch.Tensor,
        s: float = 1,
        k: float = 0
    ) -> torch.Tensor:
    """ 
    Uses Beer-Lambert law to calculate transmittance given the
    attenuation coefficients and positions of sampled points along ray.
    Uses a version of the law that has been adjusted for scaling of the
    transmittance by adding k, taking the log, and dividing by s.

    Args:
        attenuation_coeffs (torch.Tensor): shape (B, N)
        positions (torch.Tensor): shape (B, N)

    Returns:
        torch.Tensor: shape (B,)
    """
    exp = torch.exp(-torch.sum(attenuation_coeffs * distances, dim=1))
    return torch.log(exp + k) / s


@torch.no_grad()
def get_rays(
        pixel_pos: torch.Tensor, 
        angle: torch.Tensor, 
        img_shape: torch.Tensor, # TODO: does this really need to be batched?
        ) -> torch.Tensor:
    """
    Get the sampled points along the ray between the near and far bound of the image using the stratified
    sampling defined in the paper.
    
    Args:
        pixel_pos (torch.Tensor): shape (B, 2)
        angle (torch.Tensor): shape (B,)
        img_shape (torch.Tensor): shape (B, 2)
    Returns:
        torch.Tensor: shape (B * n_samples, 3)
    """

    start_pos, heading_vector = _get_start_pos(pixel_pos, angle, img_shape)
    ray_bounds = _get_ray_bounds(start_pos, heading_vector)
    
    return start_pos, heading_vector, ray_bounds


@torch.no_grad()
def get_samples(
        start_pos: torch.Tensor, 
        heading_vector: torch.Tensor, 
        ray_bounds: torch.Tensor,
        n_samples: int,
        slice_size_cm: float,
        ) -> torch.Tensor:

    t_samples = _coarse_sampling(ray_bounds, n_samples)
    sampling_distances = _get_sampling_distances(t_samples, ray_bounds, slice_size_cm)

    # sampled points should have shape (B, n_samples, 3)
    sampled_points = start_pos.unsqueeze(1) + t_samples.unsqueeze(2) * heading_vector.unsqueeze(1)
    sampled_points = sampled_points.reshape(-1, 3)

    return sampled_points, sampling_distances


@torch.no_grad()
def _get_start_pos(
        pixel_pos: torch.Tensor, 
        angle: torch.Tensor, 
        img_shape: torch.Tensor
        ) -> torch.Tensor:
    """
    Helper function to get the start position and heading vector
    for a ray in an image.
    
    Args:
        pixel_pos (torch.Tensor): shape (B, 2)
        angle (torch.Tensor): shape (B,)
        img_shape (torch.Tensor): shape (B, 2)

    Returns:
        torch.Tensor: shape (B, 3)
        torch.Tensor: shape (B, 3)
    """

    # get normalized position before accounting for angle
    normalized_pos = 2 * pixel_pos / img_shape - 1
    x = torch.ones((pixel_pos.shape[0], 1), device=pixel_pos.device) # start position is always at x = 1
    normalized_pos = torch.cat((x, normalized_pos), dim=1)

    # invert z axis
    normalized_pos[:,2] *= -1

    # rotate to account for angle
    rotation_matrix, heading_vector = _create_z_rotation_matrix(angle)
    start_pos = torch.bmm(rotation_matrix, normalized_pos.unsqueeze(2)).squeeze(2)

    return start_pos, heading_vector


@torch.no_grad()
def _create_z_rotation_matrix(angles):
    """
    Create a batch of 3D rotation matrices for rotations around the z-axis. Also returns the heading vector.
    
    Args:
        angles (torch.Tensor): A tensor of shape (B,) containing the rotation angles in radians.
        
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) containing the rotation matrices.
    """
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    rotation_matrices = torch.zeros((angles.shape[0], 3, 3), device=angles.device)
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    rotation_matrices[:, 2, 2] = 1

    heading_vector = -torch.stack((cos_angles, sin_angles, torch.zeros(angles.shape[0], device=angles.device)), dim=1)
    
    return rotation_matrices, heading_vector


@torch.no_grad()
def _get_ray_bounds(start_pos: torch.Tensor, heading_vector: torch.Tensor) -> torch.Tensor:
    """
    Given a start position (a, b, c) and a direction vector (v_x, v_y, 0), a ray can be parameterized as
    (a + t*v_x, b + t*v_y, c). This function returns the t's that correspond to the ray passing through
    the cylinder x^2 + y^2 = 1, which defines the bounds of the image. Which t corresponds to which
    bound is irrelevant since they will only be used sample points between the two t's. 

    Args:
        start_pos (torch.Tensor): shape (B, 3)
        heading_vector (torch.Tensor): shape (B, 3)

    Returns:
        torch.Tensor: A tensor of shape (B, 2), containing the two t's
    """

    return torch.stack((torch.zeros_like(start_pos[:,0]), 2*torch.ones_like(start_pos[:,0])), dim=1)

    # a = start_pos[:,0]
    # b = start_pos[:,1]

    # v_x = heading_vector[:,0]
    # v_y = heading_vector[:,1]

    # # p_half = (a*v_x + b*v_y) / (v_x**2 + v_y**2)
    # # q = (a**2 + b**2 - 1) / (v_x**2 + v_y**2)
    # p_half = (a*v_x + b*v_y)
    # q = (a**2 + b**2 - 1)
    # sq_root = torch.sqrt(p_half**2 - q)
    # sq_root = torch.nan_to_num(sq_root, nan=0) # rounding errors can cause negative values under square root

    # t1 = -p_half - sq_root
    # t2 = -p_half + sq_root

    # return torch.stack((t1, t2), dim=1)


@torch.no_grad()
def _coarse_sampling(
        ray_bounds: torch.Tensor, 
        n_samples: int, 
        ) -> torch.Tensor:
    """
    Samples n_samples points along the ray between the two t's in ray_bounds using the stratified
    sampling defined in the paper. 
    
    If proportional_sampling is True, the number of samples is
    proportional to the length of the ray, such that only rays that travel the full length of
    the image are sampled n_samples times. This reduces training time and also discourages
    the network from focusing too much on the edge of the image.

    Args:
        ray_bounds (torch.Tensor): shape (B, 2). Contains the two t bounds
        n_samples (int): number of samples
    Returns:
        torch.Tensor: shape (B, n_samples). Contains the sampled t's, and nan if there is no sample.
    """

    interval_size = (ray_bounds[:, 1] - ray_bounds[:, 0]).unsqueeze(1) / n_samples

    uniform_samples = torch.arange(0, n_samples, device=ray_bounds.device).repeat(ray_bounds.shape[0], 1)
    uniform_samples = interval_size * uniform_samples + ray_bounds[:, 0].unsqueeze(1) # This rescales each row to [t_min, t_max)

    perturbation = torch.rand(ray_bounds.shape[0], n_samples, device=ray_bounds.device) * interval_size

    return uniform_samples + perturbation


@torch.no_grad()
def _get_sampling_distances(
    t_samples: torch.Tensor, 
    ray_bounds: torch.Tensor,
    slice_size_cm: float = 0.97 * 512
    ) -> torch.Tensor:
    """
    Get the distance between sampled points along the ray. The distance associated with each sample
    is the distance between that sample and the next sample. The distance for sample sigma_i is
    t_{i+1} - t_i. The distance for the last sample is the distance between the last sample and the
    far bound of the image. 
    
    This the distance can be calculated from t since the heading vector has magnitue 1. 

    Args:
        t_samples (torch.Tensor): shape (B, n_samples)

    Returns:
        torch.Tensor: shape (B, n_samples)
    """
    far_bounds = ray_bounds[:, 1]
    sample_distances = torch.zeros(t_samples.shape[0], t_samples.shape[1], device=t_samples.device)
    sample_distances[:, :-1] = t_samples[:, 1:] - t_samples[:, :-1]
    sample_distances[:, -1] = far_bounds - t_samples[:, -1]

    # scale to cm
    sample_distances = sample_distances * slice_size_cm / 2
    return sample_distances

import torch
from dataloading import XRayDataset
from utils import get_data_dir
from torch.utils.data import DataLoader



def log_beer_lambert_law(attenuation_coeffs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """ 
    Uses Beer-Lambert law to calculate transmittance given the
    attenuation coefficients and positions of sampled points along ray.
    Uses a version of the law where the logarithm of the transmittance
    is returned, since that is what is used as ground truth for scaling reasons.

    Args:
        attenuation_coeffs (torch.Tensor): shape (B, N)
        positions (torch.Tensor): shape (B, N)

    Returns:
        torch.Tensor: shape (B,)
    """
    return -torch.sum(attenuation_coeffs * positions, dim=1)


def get_sampled_points(
        pixel_pos: torch.Tensor, 
        angle: torch.Tensor, 
        img_shape: torch.Tensor,
        n_samples: int,
        proportional_sampling: bool = False
        ) -> torch.Tensor:
    """
    Get the sampled points along the ray between the near and far bound of the image using the stratified
    sampling defined in the paper.
    
    Args:
        pixel_pos (torch.Tensor): shape (B, 2)
        angle (torch.Tensor): shape (B,)
        img_shape (torch.Tensor): shape (B, 2)
        n_samples (int): number of samples
        proportional_sampling (bool): whether to sample proportional to the length of the ray

    Returns:
        torch.Tensor: shape (B, n_samples, 3)
    """

    start_pos, heading_vector = get_start_pos(pixel_pos, angle, img_shape)
    ray_bounds = get_ray_bounds(start_pos, heading_vector)
    t_samples = stratified_sampling(ray_bounds, n_samples, proportional_sampling)

    # sampled points should have shape (B, n_samples, 3)
    sampled_points = start_pos.unsqueeze(1) + t_samples.unsqueeze(2) * heading_vector.unsqueeze(1)

    return sampled_points

def get_start_pos( # TODO: do all these computations in the dataloader so they don't have to be done over and over
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
    normalized_pos = 2 * pixel_pos / img_shape - 1 # TODO: should y-axis be flipped? So -1 is bottom and 1 is top?
    x = torch.ones((pixel_pos.shape[0], 1), device=pixel_pos.device) # start position is always at x = 1
    normalized_pos = torch.cat((x, normalized_pos), dim=1)

    # rotate to account for angle
    rotation_matrix, heading_vector = _create_z_rotation_matrix(angle)
    start_pos = torch.bmm(rotation_matrix, normalized_pos.unsqueeze(2)).squeeze(2)

    return start_pos, heading_vector


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

    heading_vector = -torch.stack((cos_angles, sin_angles, torch.zeros(angles.shape[0])), dim=1) # TODO: decide on a coordinate system. Using the same coordinate system as images in pytorch means the x-axis points away from the viewer and the z-axis points down.
    
    return rotation_matrices, heading_vector


def get_ray_bounds(start_pos: torch.Tensor, heading_vector: torch.Tensor) -> torch.Tensor:
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


    a = start_pos[:,0]
    b = start_pos[:,1]

    v_x = heading_vector[:,0]
    v_y = heading_vector[:,1]

    p_half = (a*v_x + b*v_y) / (v_x**2 + v_y**2)
    q = (a**2 + b**2 - 1) / (v_x**2 + v_y**2)

    t1 = -p_half - torch.sqrt(p_half**2 - q)
    t2 = -p_half + torch.sqrt(p_half**2 - q)

    return torch.stack((t1, t2), dim=1)


def stratified_sampling(
        ray_bounds: torch.Tensor, 
        n_samples: int, 
        proportional_sampling: bool = False
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
        start_pos (torch.Tensor): shape (B, 3)
        heading_vector (torch.Tensor): shape (B, 3)
        proportional_sampling (bool): whether to sample proportional to the length of the ray

    Returns:
        torch.Tensor: shape (B, n_samples). Contains the sampled t's, and nan if there is no sample.
    """

    samples = torch.zeros(ray_bounds.shape[0], n_samples, device=ray_bounds.device)

    if proportional_sampling:
        n_samples = n_samples * torch.abs(ray_bounds[:, 1] - ray_bounds[:, 0]) / 2
    else:
        n_samples = torch.ones(ray_bounds.shape[0], device=ray_bounds.device) * n_samples
    
    for i in range(ray_bounds.shape[0]):
        interval_size = (ray_bounds[i, 1] - ray_bounds[i, 0]) / n_samples[i]
        uniform_samples = torch.arange(ray_bounds[i, 0], ray_bounds[i, 1], interval_size, device=ray_bounds.device)
        perturbation = torch.rand(int(n_samples[i]), device=uniform_samples.device) * interval_size # uniform_samples is [lower, upper) so perturbation should be purely additive
        samples[i,:len(uniform_samples)] = uniform_samples + perturbation
        samples[i, len(uniform_samples):] = torch.nan
    return samples

    
        



def test_start_pos():
    pixel_pos = torch.tensor([[0,0],
                              [256, 256],
                              [512, 512]])
    # pixel_pos = torch.tensor([[256,256],
    #                           [512,256],
    #                           [256,512],])
    angle = torch.tensor([0,
                          3.1415926535/2,
                          3.1415926535])
    img_shape = torch.tensor([[512, 512],
                             [512, 512],
                             [512, 512]])

    start_pos, direction_vector = get_start_pos(pixel_pos, angle, img_shape)
    print(f"post rotation:\n{start_pos}\n")
    print(f"direction vector:\n{direction_vector}")

    """
    Should print: 

    post rotation:
    tensor([[ 1.0000e+00, -1.0000e+00, -1.0000e+00],
            [-4.3711e-08,  1.0000e+00,  0.0000e+00],
            [-1.0000e+00, -1.0000e+00,  1.0000e+00]])
    
    direction vector:
    tensor([[-1.0000e+00, -0.0000e+00, -0.0000e+00],
            [ 4.3711e-08, -1.0000e+00, -0.0000e+00],
            [ 1.0000e+00,  8.7423e-08, -0.0000e+00]])
    """

def test_get_samples():
    data_dir = get_data_dir() / "xrays" / "2 AC_CT_TBody"
    dataset = XRayDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    pixel_positions, angles, _ = next(iter(dataloader))
    sampled_points = get_sampled_points(pixel_positions, angles, torch.tensor([512,536]), 10)

    print(f"sampled points:\n{sampled_points}")


if __name__ == "__main__":
    test_get_samples()
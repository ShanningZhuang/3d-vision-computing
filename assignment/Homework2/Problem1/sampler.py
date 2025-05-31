import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (2): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # Get the batch size and number of rays
        batch_size = ray_bundle.origins.shape[0]
        
        # Generate uniform z values between min_depth and max_depth
        z_vals = torch.linspace(
            self.min_depth, 
            self.max_depth, 
            self.n_pts_per_ray, 
            device=ray_bundle.origins.device
        )
        # Expand z_vals to match the batch size: (batch_size, n_pts_per_ray)
        z_vals = z_vals.expand(batch_size, self.n_pts_per_ray)

        # TODO (2): Sample points from z values
        # Compute sample points along rays: origins + z_vals * directions
        # ray_bundle.origins: (batch_size, 3)
        # ray_bundle.directions: (batch_size, 3)
        # z_vals: (batch_size, n_pts_per_ray)
        
        # Expand origins and directions to match sampling dimensions
        origins = ray_bundle.origins.unsqueeze(1)  # (batch_size, 1, 3)
        directions = ray_bundle.directions.unsqueeze(1)  # (batch_size, 1, 3)
        z_vals = z_vals.unsqueeze(-1)  # (batch_size, n_pts_per_ray, 1)
        
        # Sample points: origins + z * directions
        sample_points = origins + z_vals * directions  # (batch_size, n_pts_per_ray, 3)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}
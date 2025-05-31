import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (4): Compute transmittance using the equation described in the README
        # From NeRF paper: T_i = exp(-Σ[j=1 to i-1] σ_j δ_j)
        # This is equivalent to: T_i = Π[j=1 to i-1] exp(-σ_j δ_j) = Π[j=1 to i-1] (1 - α_j)
        # where α_j = 1 - exp(-σ_j δ_j)

        # Compute alpha values: α_i = 1 - exp(-σ_i * δ_i)
        alpha = 1.0 - torch.exp(-rays_density * deltas)  # (batch, n_pts, 1)

        # Flatten channel dimension
        alpha = alpha[..., 0]  # (batch, n_pts)

        # Compute 1 - alpha plus eps
        alpha_comp = 1.0 - alpha + eps  # (batch, n_pts)

        # Prepend 1 for T_1 = 1
        ones = torch.ones((alpha_comp.shape[0], 1), device=alpha.device)  # (batch, 1)
        alpha_shifted = torch.cat([ones, alpha_comp], dim=1)  # (batch, n_pts+1)

        # Cumulative product to get T for each sample, then remove last element
        transmittance_all = torch.cumprod(alpha_shifted, dim=1)  # (batch, n_pts+1)
        transmittance = transmittance_all[:, :-1]  # (batch, n_pts)

        # Restore channel dimension
        transmittance = transmittance.unsqueeze(-1)  # (batch, n_pts, 1)

        # Compute weight: w_i = T_i * α_i
        weights = transmittance * alpha.unsqueeze(-1)  # (batch, n_pts, 1)

        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (4): Aggregate (weighted sum of) features using weights
        # L(x, ω) = Σ(wi * Le(xi, ω))
        feature = torch.sum(weights * rays_feature, dim=-2)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            )

            # TODO (4): Render (color) features using weights
            feature = self._aggregate(
                weights.view(-1, n_pts, 1),
                feature.view(-1, n_pts, 3)
            )

            # TODO (4): Render depth map
            # Depth is computed as weighted average of sample depths
            depth = self._aggregate(
                weights.view(-1, n_pts, 1),
                depth_values.view(-1, n_pts, 1)
            )

            # Add white background if specified
            if self._white_background:
                # Compute accumulated alpha (opacity)
                acc_alpha = torch.sum(weights.view(-1, n_pts, 1), dim=-2)
                feature = feature + (1.0 - acc_alpha) * torch.ones_like(feature)

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}

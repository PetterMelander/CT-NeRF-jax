import math

import torch


class XRayModel(torch.nn.Module):
    def __init__(self, n_layers: int, layer_dim: int, L: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(3 * 2 * L, layer_dim)]
            + [torch.nn.Linear(layer_dim, layer_dim) for _ in range(n_layers - 1)]
        )
        self.layers[int(len(self.layers) / 2) + 1] = torch.nn.Linear(
            layer_dim + 3 * 2 * L, layer_dim
        )
        self.out = torch.nn.Linear(layer_dim, 1)
        self.L = L

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        pos_enc = self._positional_encoding(coords, self.L)
        x = torch.clone(pos_enc)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) / 2 + 1:
                x = torch.cat((pos_enc, x), dim=1)
            x = torch.nn.functional.relu(layer(x))

        x = self.out(x)
        # if self.training:
        #     x += torch.randn_like(x, device=x.device) * 0.01
        return torch.nn.functional.leaky_relu(x)

    @torch.no_grad()
    def _positional_encoding(self, coords: torch.Tensor, L: int) -> torch.Tensor:
        """
        Positional encoding for the model
        """
        position_enc = torch.zeros(coords.shape[0], 3 * 2 * L, device=coords.device)
        for i in range(3):  # TODO: can this loop be avoided?
            position_enc[:, L * 2 * i : L * 2 * (i + 1)] = self._gamma(coords[:, i], L)
        return position_enc

    def _gamma(self, p: torch.Tensor, L: int) -> torch.Tensor:
        """
        Implements the gamma function from the original NeRF paper. The sin and cos functions
        are not ordered in the same way as in the paper, but this does not matter as they are
        input into and MLP which treats all inputs equally.
        """
        gamma = torch.zeros(p.shape[0], L * 2, device=p.device)
        angles = (
            torch.pow(2, torch.arange(L, device=p.device)).expand(p.shape[0], L)
            * math.pi
            * p.unsqueeze(1)
        )
        gamma[:, :L] = torch.sin(angles)
        gamma[:, L:] = torch.cos(angles)
        return gamma

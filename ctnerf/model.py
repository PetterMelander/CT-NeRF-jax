"""Contains the MLP model used to generate CT images."""

from typing import Any

import torch


class XRayModel(torch.nn.Module):
    """Class for the MLP used to generate CT images."""

    def __init__(
        self,
        n_layers: int,
        layer_dim: int,
        L: int,  # noqa: N803
        *args: tuple,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the XRayModel.

        Args:
            n_layers (int): Number of layers in the model
            layer_dim (int): Dimension of the layers
            L (int): Number of frequencies to use for the positional encoding
            *args: Additional positional arguments passed to the base class
            **kwargs: Additional keyword arguments passed to the base class

        """
        super().__init__(*args, **kwargs)

        self.input_layer = torch.nn.Linear(3 * 2 * L, layer_dim)
        self.pre_concat_layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_dim, layer_dim) for _ in range(n_layers // 2)],
        )
        self.middle_layer = torch.nn.Linear(layer_dim + 3 * 2 * L, layer_dim)
        self.post_concat_layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_dim, layer_dim) for _ in range(n_layers // 2 - 1)],
        )
        self.output_layer = torch.nn.Linear(layer_dim, 1)

        self.L = L

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            coords (torch.Tensor): shape (B, 3). Input coordinates.

        Returns:
            torch.Tensor: shape (B, 1). Output of the model.

        """
        pos_enc = self._positional_encoding(coords, self.L)

        x = torch.nn.functional.relu(self.input_layer(pos_enc))

        for layer in self.pre_concat_layers:
            x = torch.nn.functional.relu(layer(x))

        x = torch.cat((x, pos_enc), dim=1)
        x = torch.nn.functional.relu(self.middle_layer(x))

        for layer in self.post_concat_layers:
            x = torch.nn.functional.relu(layer(x))

        return self.output_layer(x)

        # if self.training:
        #     x += torch.randn_like(x, device=x.device) * 0.01  # noqa: ERA001
        # return torch.nn.functional.elu(x) + 1  # noqa: ERA001
        # return torch.nn.functional.relu(x)  # noqa: ERA001
        # return torch.nn.functional.gelu(x)  # noqa: ERA001
        # return torch.nn.functional.leaky_relu(x)  # noqa: ERA001
        # return torch.nn.functional.sigmoid(x)  # noqa: ERA001

    @torch.no_grad()
    def _positional_encoding(self, coords: torch.Tensor, L: int) -> torch.Tensor:  # noqa: N803
        """Compute the positional encoding for the input coordinates.

        This function applies a positional encoding to the input coordinates using
        sinusoidal functions of varying frequencies. The encoding is used to map
        the input coordinates into a higher-dimensional space, which helps the
        model to capture spatial relationships.

        Args:
            coords (torch.Tensor): A tensor of shape (B, 3) representing the input
                coordinates for which the positional encoding is to be computed.
            L (int): The number of frequency bands to use for the encoding.

        Returns:
            torch.Tensor: A tensor of shape (B, 3 * 2 * L) containing the positional
            encoding of the input coordinates.

        """
        position_enc = torch.empty(coords.shape[0], 3 * 2 * L, device=coords.device)
        angles = torch.pow(2, torch.arange(L, device=coords.device)) * torch.pi
        angles = angles.unsqueeze(0).unsqueeze(0) * coords.unsqueeze(-1)
        position_enc[:, : 3 * L] = torch.sin(angles.view(coords.shape[0], -1))
        position_enc[:, 3 * L :] = torch.cos(angles.view(coords.shape[0], -1))
        return position_enc

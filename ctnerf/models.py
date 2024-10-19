import torch


class XRayModel(torch.nn.Module):
    def __init__(self, n_layers: int, layer_dim: int, L: int, *args, **kwargs) -> None:
        """
        Initialize the XRayModel

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
            [torch.nn.Linear(layer_dim, layer_dim) for _ in range(n_layers // 2)]
        )
        self.middle_layer = torch.nn.Linear(layer_dim + 3 * 2 * L, layer_dim)
        self.post_concat_layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_dim, layer_dim) for _ in range(n_layers // 2 - 1)]
        )
        self.output_layer = torch.nn.Linear(layer_dim, 1)

        # self.layers = torch.nn.ModuleList(
        #     [torch.nn.Linear(3 * 2 * L, layer_dim)]
        #     + [torch.nn.Linear(layer_dim, layer_dim) for _ in range(n_layers - 1)]
        # )
        # self.layers[int(len(self.layers) / 2) + 1] = torch.nn.Linear(
        #     layer_dim + 3 * 2 * L, layer_dim
        # )
        # self.out = torch.nn.Linear(layer_dim, 1)
        self.L = L

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        pos_enc = self._positional_encoding(coords, self.L)

        x = torch.nn.functional.relu(self.input_layer(pos_enc))

        for layer in self.pre_concat_layers:
            x = torch.nn.functional.relu(layer(x))

        x = torch.cat((x, pos_enc), dim=1)
        x = torch.nn.functional.relu(self.middle_layer(x))

        for layer in self.post_concat_layers:
            x = torch.nn.functional.relu(layer(x))

        x = self.output_layer(x)
        return x

        # pos_enc = self._positional_encoding(coords, self.L)
        # x = torch.clone(pos_enc)
        # for i, layer in enumerate(self.layers):
        #     if i == len(self.layers) / 2 + 1:  # TODO: if-statement
        #         x = torch.cat((pos_enc, x), dim=1)
        #     x = torch.nn.functional.relu(layer(x))

        # x = self.out(x)
        # if self.training:
        #     x += torch.randn_like(x, device=x.device) * 0.01
        # return torch.nn.functional.elu(x) + 1
        # return torch.nn.functional.relu(x)
        # return torch.nn.functional.gelu(x)
        # return torch.nn.functional.leaky_relu(x)
        # return torch.nn.functional.sigmoid(x)
        return x

    @torch.no_grad()
    def _positional_encoding(self, coords: torch.Tensor, L: int) -> torch.Tensor:
        """
        Computes the positional encoding for the input coordinates.

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

    # @torch.no_grad()
    # def _positional_encoding(self, coords: torch.Tensor, L: int) -> torch.Tensor:
    #     """
    #     Positional encoding for the model
    #     """
    #     position_enc = torch.zeros(coords.shape[0], 3 * 2 * L, device=coords.device)
    #     for i in range(
    #         3
    #     ):  # TODO: can this loop be avoided? yes: angles[None, None, ...] * p[...,None]
    #         position_enc[:, L * 2 * i : L * 2 * (i + 1)] = self._gamma(coords[:, i], L)
    #     return position_enc

    # def _gamma(self, p: torch.Tensor, L: int) -> torch.Tensor:
    #     """
    #     Implements the gamma function from the original NeRF paper. The sin and cos functions
    #     are not ordered in the same way as in the paper, but this does not matter as they are
    #     input into and MLP which treats all inputs equally.
    #     """
    #     gamma = torch.zeros(p.shape[0], L * 2, device=p.device)
    #     angles = torch.pow(2, torch.arange(L, device=p.device)) * torch.pi * p.unsqueeze(1)
    #     gamma[:, :L] = torch.sin(angles)
    #     gamma[:, L:] = torch.cos(angles)
    #     return gamma

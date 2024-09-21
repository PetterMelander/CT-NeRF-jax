import torch
import math



class XRayModel(torch.nn.Module):


    def __init__(self, n_layers: int, layer_dim: int, L: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.ModuleList( # TODO: add positional encoding again midway through stack
            [torch.nn.Linear(3*2*L, layer_dim)] +
            [torch.nn.Linear(layer_dim, layer_dim) for _ in range(n_layers - 1)]
        )
        self.out = torch.nn.Linear(layer_dim, 1)
        self.L = L

    
    def forward(self, coords: torch.Tensor) -> torch.Tensor: # TODO: dropout? noise? regularization?
        x = self._positional_encoding(coords, self.L)
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        return self.out(x)
    

    def _positional_encoding(self, coords: torch.Tensor, L: int) -> torch.Tensor:
        """
        Positional encoding for the model
        """
        position_enc = torch.zeros(3*2*L, device=coords.device)
        for i in range(3):
            position_enc[L*2*i: L*2*(i+1)] = self._gamma(coords[i], L)
        return position_enc


    def _gamma(p: torch.Tensor, L: int) -> torch.Tensor:
        """
        Implements the gamma function from the original NeRF paper
        """
        gamma = torch.zeros(L*2, device=p.device)
        for i in range(L):
            gamma[2*i] = torch.sin(2**i*math.pi*p)
            gamma[2*i+1] = torch.cos(2**i*math.pi*p)
        return gamma

import torch

from mamba_ssm import Mamba
from rau.unidirectional.simple import SimpleUnidirectional


class MambaBlock(torch.nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaStackUnidirectional(SimpleUnidirectional):
    def __init__(
        self,
        d_model,
        num_layers,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = torch.nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = torch.nn.LayerNorm(d_model)

    def initial_output(self, batch_size, *args, **kwargs):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

    def forward_sequence(self, input_sequence, *args, **kwargs):
        x = input_sequence
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

    def forward_single(self, input_tensor, *args, **kwargs):
        # Fallback for RAU's single-step API.
        # Mamba is intended to run on full sequences here.
        return self.forward_sequence(input_tensor[:, None, :], *args, **kwargs).squeeze(1)

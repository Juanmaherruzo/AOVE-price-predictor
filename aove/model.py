"""Bimodal LSTM regression model for AOVE price prediction.

This is the single source of truth for the network architecture, imported by the
training, inference, CLI and API layers alike.
"""

import torch
import torch.nn as nn


class AOVEPricePredictor(nn.Module):
    """Bimodal deep learning architecture.

    Stream A — an LSTM over the high-frequency climate sequence.
    Stream B — a macro snapshot injected post-LSTM at the fusion layer.
    Both streams are concatenated and passed through a fully-connected head.
    """

    def __init__(
        self,
        hf_input_dim: int,
        macro_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=hf_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        fusion_dim = hidden_dim + macro_input_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_hf: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        """Run a forward pass over one climate window plus its macro snapshot."""
        h0 = torch.zeros(
            self.num_layers,
            x_hf.size(0),
            self.hidden_dim,
            dtype=torch.float32,
            device=x_hf.device,
        )
        c0 = torch.zeros(
            self.num_layers,
            x_hf.size(0),
            self.hidden_dim,
            dtype=torch.float32,
            device=x_hf.device,
        )
        lstm_out, _ = self.lstm(x_hf, (h0, c0))
        h_last = self.dropout(lstm_out[:, -1, :])
        fused = torch.cat([h_last, x_macro], dim=1)
        output: torch.Tensor = self.fc(fused)
        return output

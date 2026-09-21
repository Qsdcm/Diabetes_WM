"""Models used only for action-dependence baseline experiments."""

from __future__ import annotations

import torch
from torch import nn


class CGMOnlyGRU(nn.Module):
    """History CGM+Time encoder with a Time-only future rollout.

    The public forward signature matches the full model for use with the same
    training loop, but Insulin and Carb channels are deliberately ignored.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_history_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_history_layers > 1 else 0.0
        self.history_encoder = nn.GRU(
            3,
            hidden_dim,
            num_layers=num_history_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.transition = nn.GRUCell(2, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, history: torch.Tensor, future_controls: torch.Tensor
    ) -> torch.Tensor:
        if history.shape[-1] != 5 or future_controls.shape[-1] != 4:
            raise ValueError("CGM-only model expects 5-D history and 4-D future controls")
        # History channels: CGM, time_sin, time_cos. Future: time_sin, time_cos.
        cgm_time_history = torch.cat([history[..., 0:1], history[..., 3:5]], dim=-1)
        future_time = future_controls[..., 2:4]
        _, hidden = self.history_encoder(cgm_time_history)
        state = hidden[-1]
        predictions = []
        for step in range(future_time.shape[1]):
            state = self.transition(future_time[:, step], state)
            predictions.append(self.decoder(state).squeeze(-1))
        return torch.stack(predictions, dim=1)

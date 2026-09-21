"""V1 continuous-state rollout model and direct-input GRU baseline."""

from __future__ import annotations

import torch
from torch import nn


class ScalarEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ConceptScalarEncoder(ScalarEncoder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim)
        self.concept_embedding = nn.Parameter(torch.empty(embedding_dim))
        nn.init.normal_(self.concept_embedding, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value) + self.concept_embedding


class TimeEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class WorldModelV1(nn.Module):
    """Encode history into a state, then roll it forward using controls only."""

    def __init__(
        self,
        *,
        embedding_dim: int = 16,
        event_dim: int = 64,
        hidden_dim: int = 128,
        num_history_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cgm_encoder = ScalarEncoder(embedding_dim)
        self.insulin_encoder = ConceptScalarEncoder(embedding_dim)
        self.carb_encoder = ConceptScalarEncoder(embedding_dim)
        self.time_encoder = TimeEncoder(embedding_dim)
        self.history_fusion = nn.Sequential(
            nn.Linear(4 * embedding_dim, event_dim), nn.LayerNorm(event_dim), nn.SiLU()
        )
        self.future_fusion = nn.Sequential(
            nn.Linear(3 * embedding_dim, event_dim), nn.LayerNorm(event_dim), nn.SiLU()
        )
        gru_dropout = dropout if num_history_layers > 1 else 0.0
        self.history_encoder = nn.GRU(
            event_dim,
            hidden_dim,
            num_layers=num_history_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        # This one transition cell is reused at every rollout step.
        self.transition = nn.GRUCell(event_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-1] != 5:
            raise ValueError("history must contain [CGM, Insulin, Carb, time_sin, time_cos]")
        cgm = self.cgm_encoder(history[..., 0:1])
        insulin = self.insulin_encoder(history[..., 1:2])
        carb = self.carb_encoder(history[..., 2:3])
        time = self.time_encoder(history[..., 3:5])
        events = self.history_fusion(torch.cat([cgm, insulin, carb, time], dim=-1))
        _, hidden = self.history_encoder(events)
        return hidden[-1]

    def encode_future_event(self, future_control: torch.Tensor) -> torch.Tensor:
        if future_control.shape[-1] != 4:
            raise ValueError("future controls must contain [Insulin, Carb, time_sin, time_cos]")
        insulin = self.insulin_encoder(future_control[..., 0:1])
        carb = self.carb_encoder(future_control[..., 1:2])
        time = self.time_encoder(future_control[..., 2:4])
        return self.future_fusion(torch.cat([insulin, carb, time], dim=-1))

    def forward(
        self, history: torch.Tensor, future_controls: torch.Tensor
    ) -> torch.Tensor:
        state = self.encode_history(history)
        predictions = []
        for step in range(future_controls.shape[1]):
            event = self.encode_future_event(future_controls[:, step])
            state = self.transition(event, state)
            predictions.append(self.decoder(state).squeeze(-1))
        return torch.stack(predictions, dim=1)


class DirectGRUBaseline(nn.Module):
    """Direct 5-D history input with a control-only recurrent rollout."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_history_layers: int = 1,
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_history_layers > 1 else 0.0
        self.history_encoder = nn.GRU(
            5,
            hidden_dim,
            num_layers=num_history_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.transition = nn.GRUCell(4, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, history: torch.Tensor, future_controls: torch.Tensor
    ) -> torch.Tensor:
        if history.shape[-1] != 5 or future_controls.shape[-1] != 4:
            raise ValueError("baseline expects 5-D history and 4-D future controls")
        _, hidden = self.history_encoder(history)
        state = hidden[-1]
        predictions = []
        for step in range(future_controls.shape[1]):
            state = self.transition(future_controls[:, step], state)
            predictions.append(self.decoder(state).squeeze(-1))
        return torch.stack(predictions, dim=1)


def build_model(name: str, **kwargs: object) -> nn.Module:
    if name == "world_model":
        return WorldModelV1(**kwargs)
    if name == "baseline":
        return DirectGRUBaseline(**kwargs)
    raise ValueError(f"Unknown model: {name}")

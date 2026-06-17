from __future__ import annotations

import torch
from torch import nn


class CharRNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        cell: str = "gru",
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cell = cell.lower()
        if cell not in {"rnn", "gru", "lstm"}:
            raise ValueError("--cell must be one of: rnn, gru, lstm")

        self.cell = cell
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if cell == "rnn":
            self.rnn = nn.RNN(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                nonlinearity="tanh",
                batch_first=True,
            )
        elif cell == "gru":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        logits = self.output(out)
        return logits, hidden

    def config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "cell": self.cell,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }


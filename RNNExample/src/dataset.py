from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Vocabulary:
    chars: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def build(cls, text: str) -> "Vocabulary":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(chars=chars, stoi=stoi, itos=itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @property
    def size(self) -> int:
        return len(self.chars)

    def to_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocabulary":
        chars = list(payload["chars"])
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(chars=chars, stoi=stoi, itos=itos)


def load_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    if len(text) < 100:
        raise ValueError(f"Corpus is too short: {path}")
    return text


def split_data(ids: list[int], train_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(ids, dtype=torch.long)
    split = max(1, int(len(data) * train_ratio))
    return data[:split], data[split:]


def get_batch(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    split: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    if len(data) <= seq_len + 1:
        data = train_data
    if len(data) <= seq_len + 1:
        raise ValueError("Corpus is too short for the chosen --seq-len.")

    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in starts])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


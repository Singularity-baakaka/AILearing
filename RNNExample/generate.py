from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from src.dataset import Vocabulary
from src.model import CharRNNLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained RNN poet.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--prompt", default="春风")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--length", type=int, default=120)
    return parser.parse_args()


@torch.no_grad()
def generate(model, vocab, prompt, max_new_tokens, temperature, device):
    model.eval()
    ids = vocab.encode(prompt)
    if not ids:
        ids = [0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = None

    for _ in range(max_new_tokens):
        logits, hidden = model(x[:, -1:], hidden)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return vocab.decode(x[0].tolist())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocab = Vocabulary.from_dict(checkpoint["vocab"])
    model = CharRNNLanguageModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    print(generate(model, vocab, args.prompt, args.length, args.temperature, device))


if __name__ == "__main__":
    main()


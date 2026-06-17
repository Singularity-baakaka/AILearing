from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.dataset import Vocabulary, get_batch, load_text, split_data
from src.model import CharRNNLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a character-level RNN poet.")
    parser.add_argument("--corpus", default="data/sample_poetry.txt")
    parser.add_argument("--cell", choices=["rnn", "gru", "lstm"], default="gru")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", default="春风")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--log-path", default="outputs/loss.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device, criterion, eval_iters=10):
    model.eval()
    result = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(train_data, val_data, split, args.batch_size, args.seq_len, device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


@torch.no_grad()
def generate(model, vocab, prompt, max_new_tokens, temperature, device):
    was_training = model.training
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

    text = vocab.decode(x[0].tolist())
    if was_training:
        model.train()
    return text


def save_checkpoint(path, model, vocab, args, step, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "vocab": vocab.to_dict(),
            "step": step,
            "val_loss": val_loss,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    text = load_text(args.corpus)
    vocab = Vocabulary.build(text)
    train_data, val_data = split_data(vocab.encode(text))

    model = CharRNNLanguageModel(
        vocab_size=vocab.size,
        cell=args.cell,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    out_dir = Path(args.out_dir)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = math.inf

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "perplexity"])

        for step in range(1, args.steps + 1):
            x, y = get_batch(train_data, val_data, "train", args.batch_size, args.seq_len, device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab.size), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % args.eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data, args, device, criterion)
                ppl = math.exp(min(losses["val"], 20))
                writer.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{ppl:.2f}"])
                f.flush()
                print(
                    f"step {step:5d} | train {losses['train']:.4f} | "
                    f"val {losses['val']:.4f} | ppl {ppl:.2f}"
                )

                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(out_dir / "best.pt", model, vocab, args, step, best_val)

            if step == 1 or step % args.sample_interval == 0:
                sample = generate(model, vocab, args.prompt, 80, args.temperature, device)
                print("\n--- sample ---")
                print(sample)
                print("--------------\n")

    save_checkpoint(out_dir / "last.pt", model, vocab, args, args.steps, best_val)
    print(f"saved: {out_dir / 'best.pt'}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()

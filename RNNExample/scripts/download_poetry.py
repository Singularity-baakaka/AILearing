from __future__ import annotations

import argparse
import json
from pathlib import Path


POETRY_DIRS = [
    "全唐诗",
    "宋词",
    "宋诗",
    "五代诗词",
    "楚辞",
    "诗经",
    "论语",
    "四书五经",
    "元曲",
    "纳兰性德",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean chinese-poetry JSON files into a plain text corpus."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a local chinese-poetry repository.",
    )
    parser.add_argument("--output", default="data/poetry_full.txt")
    parser.add_argument("--limit", type=int, default=20000, help="Maximum poems to export.")
    return parser.parse_args()


def iter_json_files(source: Path):
    for dirname in POETRY_DIRS:
        root = source / dirname
        if root.exists():
            yield from root.rglob("*.json")
    json_root = source / "json"
    if json_root.exists():
        yield from json_root.rglob("*.json")


def extract_lines(record: dict) -> list[str]:
    paragraphs = (
        record.get("paragraphs")
        or record.get("strains")
        or record.get("content")
        or record.get("chapter")
        or record.get("rhythmic")
        or []
    )
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    lines = []
    for item in paragraphs:
        text = str(item).strip()
        if text:
            lines.append(text)
    return lines


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(
            "Source path does not exist. Download chinese-poetry first:\n"
            "  git clone https://github.com/chinese-poetry/chinese-poetry.git"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as out:
        for json_file in iter_json_files(source):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                payload = json.loads(json_file.read_text(encoding="utf-8-sig"))

            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                if not isinstance(record, dict):
                    continue
                lines = extract_lines(record)
                if not lines:
                    continue
                out.write("\n".join(lines))
                out.write("\n\n")
                count += 1
                if count >= args.limit:
                    print(f"wrote {count} poems to {output}")
                    return

    print(f"wrote {count} poems to {output}")


if __name__ == "__main__":
    main()

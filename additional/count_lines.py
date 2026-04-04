from __future__ import annotations

import sys
from pathlib import Path


TARGET_SUFFIXES = {".py", ".kt"}


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file)


def main() -> int:
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    target_dir = target_dir.expanduser().resolve()

    if not target_dir.exists():
        print(f"Directory does not exist: {target_dir}", file=sys.stderr)
        return 1

    if not target_dir.is_dir():
        print(f"Path is not a directory: {target_dir}", file=sys.stderr)
        return 1

    files = sorted(
        path for path in target_dir.rglob("*") if path.is_file() and path.suffix.lower() in TARGET_SUFFIXES
    )

    if not files:
        print(f"No .py or .kt files found in: {target_dir}")
        return 0

    total = 0
    skipped = 0
    for path in files:
        relative_path = path.relative_to(target_dir)
        try:
            line_count = count_lines(path)
        except OSError as exc:
            skipped += 1
            print(f"{relative_path}: skipped ({exc.strerror or exc.__class__.__name__})", file=sys.stderr)
            continue

        total += line_count
        print(f"{relative_path}: {line_count}")

    print(f"Total: {total}")
    if skipped:
        print(f"Skipped: {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

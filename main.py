from __future__ import annotations

import sys
from pathlib import Path

# The portable VapourSynth python ships a python313._pth that pins sys.path
# and drops even the script directory; make the project root importable.
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def print_help() -> None:
    print(
        "Usage:\n"
        "  python main.py [file|dir|plan ...]\n"
        "  python main.py extract [--overwrite] [file|dir|plan ...]\n"
        "  python main.py attachments [--outdir PATH] [file|dir|plan ...]\n"
        "  python main.py manage [workdir|dir|plan ...]\n"
        "  python main.py help"
    )


def main(argv: list[str]) -> int:
    if argv and argv[0].strip().lower() in ("-h", "--help", "help"):
        print_help()
        return 0
    # main_functions pulls GUI deps (tkinter) that the portable VS python
    # lacks; import lazily so `manage` works in both interpreters.
    if not argv:
        from main_functions import run_main

        return run_main([])

    command = argv[0].strip().lower()
    if command == "manage":
        from utils.manage.app import run_manage

        return run_manage(argv[1:])
    from main_functions import run_extract, run_extract_attachments, run_main

    if command == "extract":
        overwrite = any(arg.strip().lower() == "--overwrite" for arg in argv[1:])
        paths = [arg for arg in argv[1:] if arg.strip().lower() != "--overwrite"]
        return run_extract(paths, overwrite=overwrite)
    if command in ("attachments", "extract-attachments"):
        outdir = ""
        paths: list[str] = []
        idx = 1
        while idx < len(argv):
            token = argv[idx]
            if token == "--outdir":
                idx += 1
                if idx >= len(argv):
                    raise RuntimeError("--outdir requires a value")
                outdir = argv[idx]
            else:
                paths.append(token)
            idx += 1
        if not paths:
            paths = argv[1:] if not outdir else []
        return run_extract_attachments(paths, outdir=outdir)
    if command in ("split", "modify", "pack"):
        raise NotImplementedError(f"{command} is not implemented yet")
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

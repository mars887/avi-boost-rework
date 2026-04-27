from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_batch_manager_module() -> Any:
    path = ROOT / "utils" / "batch-manager.py"
    spec = importlib.util.spec_from_file_location("pbbatch_batch_manager", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pbbatch_batch_manager"] = module
    spec.loader.exec_module(module)
    return module


async def run_batch_tool(source_dir: str, tool: str, pass_name: str = "") -> str:
    def work() -> str:
        module = load_batch_manager_module()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            groups, unknown = module.collect_sources([source_dir])
            if unknown:
                print("Unknown inputs:")
                for item in unknown:
                    print(f"  {item}")
            if not groups:
                print("No sources found.")
                return buf.getvalue()
            for group in groups:
                print()
                print(f"[{tool}] {group.base}")
                if tool == "verify":
                    module.verify_config(group, check_filters=True, check_params=False)
                elif tool == "make_web_mp4":
                    module.make_web_mp4(group)
                elif tool == "analytics":
                    selected = [pass_name] if pass_name in ("fastpass", "mainpass") else ["fastpass", "mainpass"]
                    for selected_pass in selected:
                        module.run_pass_analytics(group, selected_pass)
                elif tool == "config_dump":
                    module.config_dump(groups)
                    break
                else:
                    print(f"Unsupported tool: {tool}")
        return buf.getvalue()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, work)


async def run_batch_edit_text(
    source_dir: str,
    *,
    target: str,
    find_text: str,
    replacement: str,
    selection: str,
) -> str:
    def work() -> str:
        module = load_batch_manager_module()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            groups, unknown = module.collect_sources([source_dir])
            if unknown:
                print("Unknown inputs:")
                for item in unknown:
                    print(f"  {item}")
            if not groups:
                print("No sources found.")
                return buf.getvalue()
            matches = module.find_edit_matches(groups, target, find_text)
            if not matches:
                print(f"[skip] no matches found in {module.edit_target_label(target)}.")
                return buf.getvalue()
            module.print_edit_matches(matches)
            selected_ids = module.enter_numbers(selection or "*", 1, len(matches))
            selected = [m for m in matches if m.index in set(selected_ids)]
            if not selected:
                print("[skip] nothing selected.")
                return buf.getvalue()
            changed = module.replace_selected_lines(selected, replacement)
            print(f"[done] Edit completed, replaced {changed} line(s).")
        return buf.getvalue()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, work)

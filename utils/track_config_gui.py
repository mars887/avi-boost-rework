import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.track_gui_app import TrackConfigGui, main
from utils.track_gui_shared import (
    DefaultSettings,
    align_pipe_table,
    build_default_defaults_dict,
    build_results,
    format_params_for_display,
    load_gui_data_from_paths,
    normalize_encoder,
    normalize_type,
    sanitize_params_display_text,
)

__all__ = [
    "TrackConfigGui",
    "DefaultSettings",
    "align_pipe_table",
    "build_default_defaults_dict",
    "build_results",
    "format_params_for_display",
    "load_gui_data_from_paths",
    "normalize_encoder",
    "normalize_type",
    "sanitize_params_display_text",
    "main",
]


if __name__ == "__main__":
    main()

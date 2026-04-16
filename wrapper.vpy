from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.vpy_wrapper_runtime import build_clip

clip = build_clip(globals())
clip.set_output()

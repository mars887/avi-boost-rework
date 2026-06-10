import sys
from pathlib import Path

# The portable python distribution pins sys.path via python313._pth and does
# not include the current directory, so make the project root importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

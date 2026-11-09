import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the guardbench log directory and credential store out of the checkout.
os.environ.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))

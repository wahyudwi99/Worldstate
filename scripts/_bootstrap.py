"""Allow direct script execution from an uninstalled source checkout."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Keep generated Matplotlib cache inside the project output directory.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))


def run(main):
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

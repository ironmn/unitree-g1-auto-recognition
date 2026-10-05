"""Compatibility entry point; implementation lives in unitree_vision."""

import sys
from pathlib import Path

# Keep existing script commands working before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unitree_vision.state import BODY, FINGERS, Joint, StateReader, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())

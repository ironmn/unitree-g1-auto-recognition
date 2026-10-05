"""Compatibility entry point; implementation lives in unitree_vision."""

import sys
from pathlib import Path

# Keep existing script commands working before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unitree_vision.collector import ObservationCollector, main, parser  # noqa: E402,F401
from unitree_vision.dataset import DatasetWriter
from unitree_vision.pairing import PairBuffer
from unitree_vision.protocol import IndexedDecoder
from unitree_vision.streams import IndexedCameraStream

if __name__ == "__main__":
    raise SystemExit(main())

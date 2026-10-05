"""Compatibility entry point; implementation lives in unitree_vision."""

import sys
from pathlib import Path

# Keep existing script commands working before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unitree_vision.camera import (  # noqa: E402,F401
    ROLES,
    CameraStream,
    PreviewSession,
    check_response,
    main,
    resolve_camera,
    tile,
    unpack_message,
)

if __name__ == "__main__":
    raise SystemExit(main())

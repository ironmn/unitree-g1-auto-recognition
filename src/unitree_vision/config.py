"""Validated collection settings; TOML is portable and needs no extra parser."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class CollectorConfig:
    grpc_address: str = "localhost:50051"
    host: str = "localhost"
    agent: str = "g1_pick"
    cameras: str = "head,wrist_r"
    head_name: str | None = None
    wrist_r_name: str | None = None
    wrist_l_name: str | None = None
    profile: str = "full"
    prompt: str = "识别面板上的按钮4"
    fps: float = 10
    timeout: float = 20
    pair_timeout: float = 3
    duration: float = 0
    record: bool = False
    sample_every: int = 1
    headless: bool = False
    output: Path = Path("data/observations")
    pair_capacity: int = 64
    camera_buffer: int = 16
    receive_only: bool = False

    def __post_init__(self):
        for key in ("grpc_address", "host", "agent", "cameras", "profile", "prompt"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        for key in ("head_name", "wrist_r_name", "wrist_l_name"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{key} must be a nonempty string or None")
        roles = self.cameras.split(",")
        if len(set(roles)) != len(roles) or any(
            r not in ("head", "wrist_r", "wrist_l") for r in roles
        ):
            raise ValueError("cameras must contain unique head,wrist_r,wrist_l roles")
        if self.profile not in ("full", "arms", "right"):
            raise ValueError("profile must be full, arms or right")
        for key in ("fps", "timeout", "pair_timeout", "duration"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
        if not (
            1 <= self.fps <= 30
            and self.timeout > 0
            and self.pair_timeout > 0
            and self.duration >= 0
        ):
            raise ValueError("fps in [1,30], timeouts > 0, duration >= 0")
        for key in ("sample_every", "pair_capacity", "camera_buffer"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("record", "headless", "receive_only"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if not isinstance(self.output, (str, Path)) or not str(self.output).strip():
            raise ValueError("output must be a nonempty path")
        object.__setattr__(self, "output", Path(self.output))

    def snapshot(self):
        return {**asdict(self), "output": str(self.output)}

    @classmethod
    def from_args(cls, args):
        if isinstance(args, cls):
            return args
        return cls(**{f.name: getattr(args, f.name, f.default) for f in fields(cls)})


def parse_config(parser, argv=None):
    """CLI overrides file settings; unknown fields fail before any connection."""
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        try:
            with Path(preliminary.config).open("rb") as file:
                document = tomllib.load(file)
            if set(document) != {"collector"} or not isinstance(document["collector"], dict):
                raise ValueError("config must contain exactly one [collector] table")
            values = document["collector"]
            allowed = {f.name for f in fields(CollectorConfig)} - {"receive_only"}
            if unknown := set(values) - allowed:
                raise ValueError(f"unknown collector fields: {sorted(unknown)}")
            CollectorConfig(**values)  # Validate TOML types before argparse conversions.
            parser.set_defaults(**values)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    try:
        config = CollectorConfig.from_args(parser.parse_args(argv))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if config.headless and not config.record:
        parser.error("headless mode requires --record")
    return config

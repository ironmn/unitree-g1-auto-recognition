"""Small Windows client and logged action guards for a local G1 policy service."""

from __future__ import annotations

import dataclasses
import io
import json
import math
import urllib.request

import numpy as np

IMAGE_KEYS = ("observation.images.cam_head", "observation.images.cam_wrist_r")


class PolicyClient:
    def __init__(self, url="http://127.0.0.1:8766", timeout=120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=5) as response:
            payload = response.read(1_048_577)
        if len(payload) > 1_048_576:
            raise ValueError("Oversized health response")
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError("Expected health object")
        return result

    def infer(self, state, prompt, images, seed):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (18,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 18D state")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Expected nonempty prompt")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Expected uint32 seed")
        values = {"state": state, "prompt": np.asarray(prompt), "seed": np.uint32(seed)}
        for key in IMAGE_KEYS:
            value = np.asarray(images[key])
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
                raise ValueError("Expected uint8 RGB HWC images")
            if not 1 <= value.shape[0] <= 4096 or not 1 <= value.shape[1] <= 4096:
                raise ValueError("Invalid image dimensions")
            values[key] = value
        stream = io.BytesIO()
        np.savez_compressed(stream, **values)
        request = urllib.request.Request(
            self.url + "/infer",
            data=stream.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read(1_048_577)
        if len(payload) > 1_048_576:
            raise ValueError("Oversized action response")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            actions = np.asarray(archive["actions"], dtype=np.float32)
            timing = np.asarray(archive["sampling_ms"])
        if actions.shape != (24, 18) or not np.isfinite(actions).all():
            raise ValueError("Expected finite 24x18 absolute pose/claw actions")
        if timing.shape != () or not np.isfinite(timing) or float(timing) < 0:
            raise ValueError("Invalid sampling_ms")
        return actions, float(timing)


@dataclasses.dataclass(frozen=True)
class ActionLimits:
    position_step_m: float = 0.02
    orientation_step_deg: float = 10.0
    reject_position_jump_m: float = 0.5
    absolute_radius_m: float = 1.5

    def __post_init__(self):
        for value in dataclasses.asdict(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Action limits must be finite and positive")
        if self.position_step_m >= self.reject_position_jump_m:
            raise ValueError("Position step limit must be smaller than rejection limit")
        if self.orientation_step_deg > 180:
            raise ValueError("Orientation step limit must be <=180 degrees")


def unit_quaternion(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError("Invalid xyzw quaternion")
    return q / np.linalg.norm(q)


def quaternion_angle(q1, q2):
    a, b = unit_quaternion(q1), unit_quaternion(q2)
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))


def guarded_right_action(raw_action, previous, limits=ActionLimits()):
    """Return a pose/claw command plus every applied constraint; never motor angles.

    Only right-hand fields are executed. Left fields are copied from previous,
    consistent with the official right-arm-only demonstration controller.
    Gross position jumps/invalid quaternions fail the episode instead of silently
    reusing an old prediction. Ordinary right pose increments are rate limited.
    """
    raw = np.asarray(raw_action, dtype=np.float64)
    current = np.asarray(previous, dtype=np.float64)
    if raw.shape != (18,) or current.shape != (18,):
        raise ValueError("Expected 18D actions/state")
    if not np.isfinite(raw).all() or not np.isfinite(current).all():
        raise ValueError("Non-finite action/state")
    command = current.copy()
    target_q = unit_quaternion(raw[10:14])
    current_q = unit_quaternion(current[10:14])
    displacement = raw[7:10] - current[7:10]
    distance = float(np.linalg.norm(displacement))
    if np.linalg.norm(raw[7:10]) > limits.absolute_radius_m:
        raise ValueError("Right target exceeds base-frame workspace radius")
    if distance > limits.reject_position_jump_m:
        raise ValueError("Right target exceeds pose-jump rejection limit")
    command[7:10] = current[7:10] + displacement * min(
        1.0, limits.position_step_m / max(distance, 1e-12)
    )
    dot = float(np.dot(current_q, target_q))
    if dot < 0:
        target_q, dot = -target_q, -dot
    theta = float(np.arccos(np.clip(dot, 0, 1)))
    angle = 2 * theta
    maximum = math.radians(limits.orientation_step_deg)
    if angle > maximum:
        fraction = maximum / angle
        # Unit-quaternion SLERP along the shortest rotation, robust to q/-q.
        target_q = (
            np.sin((1 - fraction) * theta) * current_q + np.sin(fraction * theta) * target_q
        ) / np.sin(theta)
    command[10:14] = unit_quaternion(target_q)
    command[16:18] = np.clip(raw[16:18], 0, 1)
    details = {
        "position_limited": distance > limits.position_step_m,
        "orientation_limited": angle > maximum,
        "gripper_clipped": bool(np.any((raw[16:18] < 0) | (raw[16:18] > 1))),
        "quaternion_normalized": abs(float(np.linalg.norm(raw[10:14])) - 1) > 1e-6,
        "raw_position_jump_m": distance,
        "raw_orientation_jump_deg": math.degrees(angle),
        "position_projection_m": float(np.linalg.norm(command[7:10] - raw[7:10])),
        "orientation_projection_deg": math.degrees(quaternion_angle(command[10:14], raw[10:14])),
        "ignored_left_position_difference_m": float(np.linalg.norm(raw[:3] - current[:3])),
    }
    return command.astype(np.float32), details


def right_gripper_motor(action, ranges=((-1.0, 2.0), (-1.0, 2.0))):
    bounds = np.asarray(ranges, dtype=np.float32)
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise ValueError("Invalid right gripper ranges")
    return np.clip(np.asarray(action)[16:18], 0, 1) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

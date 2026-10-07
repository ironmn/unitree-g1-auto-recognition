"""Offline metrics for the recorded 18D absolute G1 action contract."""

import hashlib

import numpy as np

ACTION_NAMES = [
    "l_pos_x",
    "l_pos_y",
    "l_pos_z",
    "l_quat_x",
    "l_quat_y",
    "l_quat_z",
    "l_quat_w",
    "r_pos_x",
    "r_pos_y",
    "r_pos_z",
    "r_quat_x",
    "r_quat_y",
    "r_quat_z",
    "r_quat_w",
    "l_grip_inner_norm",
    "l_grip_outer_norm",
    "r_grip_inner_norm",
    "r_grip_outer_norm",
]


def window_starts(length, horizon, stride):
    """Unique full windows, including the final full window exactly once."""
    if horizon < 1 or stride < 1 or length < horizon:
        raise ValueError("Expected a positive horizon/stride and at least one full window")
    last = length - horizon
    return sorted(set([*range(0, last + 1, stride), last]))


def stable_window_id(episode_key, frame, namespace):
    text = f"{namespace}:{episode_key}:{frame}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def action_metrics(prediction, target):
    """Position distances in mm, SO(3) angles in degrees, raw gripper errors.

    Predictions are never clipped. Invalid predicted quaternions receive a 180
    degree penalty and an explicit invalid rate. Invalid labels are rejected.
    """
    pred, true = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if pred.ndim != 2 or pred.shape != true.shape or pred.shape[1] != 18:
        raise ValueError("Expected matching H x 18 predictions and labels")
    if not len(pred) or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Non-finite or empty action")
    distances, angles, invalid, norm_error = [], [], [], []
    for pos, quat in [(slice(0, 3), slice(3, 7)), (slice(7, 10), slice(10, 14))]:
        distances.append(np.linalg.norm(pred[:, pos] - true[:, pos], axis=-1) * 1000)
        pq, tq = pred[:, quat], true[:, quat]
        pn, tn = np.linalg.norm(pq, axis=-1), np.linalg.norm(tq, axis=-1)
        if np.any(tn < 1e-8):
            raise ValueError("Invalid ground-truth quaternion")
        bad = pn < 1e-8
        dot = np.sum((pq / np.maximum(pn[:, None], 1e-8)) * (tq / tn[:, None]), axis=-1)
        angle = np.degrees(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))
        angle[bad] = 180
        angles.append(angle)
        invalid.append(bad)
        norm_error.append(np.abs(pn - 1))
    position, orientation = np.stack(distances, -1), np.stack(angles, -1)
    gripper = np.abs(pred[:, 14:18] - true[:, 14:18])
    result = {}
    for name, array in [
        ("position_mm", position),
        ("orientation_deg", orientation),
        ("gripper_mae", gripper),
    ]:
        for label, count in [("first", 1), ("first6", min(6, len(pred))), ("horizon", len(pred))]:
            result[f"{name}_{label}"] = float(array[:count].mean())
    result.update(
        {
            "left_position_mm_horizon": float(position[:, 0].mean()),
            "right_position_mm_horizon": float(position[:, 1].mean()),
            "left_orientation_deg_horizon": float(orientation[:, 0].mean()),
            "right_orientation_deg_horizon": float(orientation[:, 1].mean()),
            "quaternion_invalid_fraction": float(np.mean(invalid)),
            "quaternion_norm_mae": float(np.mean(norm_error)),
            "gripper_out_of_range_fraction": float(
                np.mean((pred[:, 14:18] < 0) | (pred[:, 14:18] > 1))
            ),
        }
    )
    return result


def equal_mean(rows):
    if not rows:
        raise ValueError("No rows to aggregate")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Inconsistent metric keys")
    return {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}


def summarize_episodes(episodes):
    """Episode means -> task means -> equal-weight three-task macro mean."""
    tasks = {}
    for task in sorted({episode["task"] for episode in episodes}):
        items = [ep for ep in episodes if ep["task"] == task]
        tasks[task] = {
            "episodes": len(items),
            "loss_windows": sum(ep["loss_windows"] for ep in items),
            "action_windows": sum(ep["action_windows"] for ep in items),
            "metrics": equal_mean([ep["metrics"] for ep in items]),
            "hold_current_state": equal_mean([ep["hold_current_state"] for ep in items]),
        }
    return {
        "tasks": tasks,
        "macro": equal_mean([v["metrics"] for v in tasks.values()]),
        "hold_current_state_macro": equal_mean([v["hold_current_state"] for v in tasks.values()]),
    }


def paired_comparison(baseline, finetuned):
    """Require identical plans, then report paired episode-level changes."""
    if baseline["plan_sha256"] != finetuned["plan_sha256"]:
        raise ValueError("Evaluation plans differ")
    before = {ep["key"]: ep for ep in baseline["episodes"]}
    after = {ep["key"]: ep for ep in finetuned["episodes"]}
    if set(before) != set(after):
        raise ValueError("Episode sets differ")
    paired = []
    for key in sorted(before):
        a, b = before[key], after[key]
        if (a["task"], a["loss_windows"], a["action_windows"]) != (
            b["task"],
            b["loss_windows"],
            b["action_windows"],
        ):
            raise ValueError("Episode sampling differs")
        paired.append(
            {
                "key": key,
                "task": a["task"],
                "delta": {k: b["metrics"][k] - v for k, v in a["metrics"].items()},
            }
        )

    def changes(a, b):
        return {
            k: {
                "baseline": v,
                "finetuned": b[k],
                "delta": b[k] - v,
                "relative_change_percent": (b[k] - v) / v * 100 if abs(v) > 1e-12 else None,
            }
            for k, v in a.items()
        }

    return {
        "macro": changes(baseline["summary"]["macro"], finetuned["summary"]["macro"]),
        "tasks": {
            task: changes(row["metrics"], finetuned["summary"]["tasks"][task]["metrics"])
            for task, row in baseline["summary"]["tasks"].items()
        },
        "paired_episodes": paired,
        "interpretation": "Negative delta means lower error/loss. No robot success-rate claim.",
    }

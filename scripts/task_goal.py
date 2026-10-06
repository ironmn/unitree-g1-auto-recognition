"""Explicit task goals and deterministic grounding against supplied detections.

This is not an OCR/detector. Catalog image detections must have provenance.
Pixel boxes identify objects, never a 3D contact point or robot command.
"""

from __future__ import annotations

import hashlib
import json
import math
import re


class GoalError(ValueError):
    pass


def bbox(value):
    if len(value) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value):
        raise GoalError("bbox must be four finite normalized xyxy coordinates")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise GoalError("bbox outside normalized image or empty")
    return list(value)


def validate_catalog(catalog):
    if catalog.get("schema_version") != 1:
        raise GoalError("Unsupported catalog version")
    ids = [x["id"] for x in catalog["targets"]]
    if len(ids) != len(set(ids)) or not ids:
        raise GoalError("Catalog needs unique instance IDs")
    for target in catalog["targets"]:
        if target["type"] not in ("push_button", "rotary", "toggle"):
            raise GoalError("Unsupported target type")
        for det in target.get("detections", []):
            bbox(det["bbox_xyxy_norm"])
            if not det.get("image_id") or not det.get("camera") or not det.get("source"):
                raise GoalError("Detection needs image identity, camera, and provenance")
            if not 0 <= det["confidence"] <= 1:
                raise GoalError("Invalid confidence")
    return catalog


def resolve_goal(request, catalog):
    """Single positive instruction or same-image ROI -> unique immutable target ID."""
    validate_catalog(catalog)
    if request.get("schema_version") != 1:
        raise GoalError("Unsupported request version")
    mode = request.get("mode")
    candidates = catalog["targets"]
    if request.get("cabinet_id"):
        candidates = [t for t in candidates if t["cabinet_id"] == request["cabinet_id"]]
    operation = request.get("operation")
    evidence = {}
    if mode == "text":
        text = request.get("text", "").strip()
        # Complex/negative instructions need a richer parser, not unsafe substring guesses.
        if any(
            s in text
            for s in [
                "不要",
                "不能",
                "禁止",
                "别按",
                "不按",
                "除了",
                "而不是",
                "先",
                "然后",
                "再按",
            ]
        ):
            raise GoalError("Only one positive operation is supported; split complex instructions")
        verbs = {
            "press": ["按压", "按下", "按一下", "按动"],
            "rotate": ["旋转", "转动"],
            "toggle": ["拨动"],
        }
        found = [op for op, words in verbs.items() if any(w in text for w in words)]
        if len(found) > 1 or (found and operation and operation != found[0]):
            raise GoalError("Conflicting operations")
        operation = operation or (found[0] if found else None)
        candidates = [
            t for t in candidates if any(a in text for a in [t["name"], *t.get("aliases", [])])
        ]
        evidence = {"source": "catalog_alias_match", "text": text}
    elif mode == "roi":
        roi = bbox(request["bbox_xyxy_norm"])
        if not request.get("image_id") or not request.get("camera"):
            raise GoalError("ROI requires image_id and camera")
        matches = []
        for target in candidates:
            for det in target.get("detections", []):
                if (
                    det["image_id"] != request["image_id"]
                    or det["camera"] != request["camera"]
                    or det["confidence"] < 0.8
                ):
                    continue
                x1, y1, x2, y2 = bbox(det["bbox_xyxy_norm"])
                area = (x2 - x1) * (y2 - y1)
                overlap = (
                    max(0, min(x2, roi[2]) - max(x1, roi[0]))
                    * max(0, min(y2, roi[3]) - max(y1, roi[1]))
                    / area
                )
                if (
                    roi[0] <= (x1 + x2) / 2 <= roi[2]
                    and roi[1] <= (y1 + y2) / 2 <= roi[3]
                    and overlap >= 0.5
                ):
                    matches.append((target, det))
                    break
        candidates = [t for t, d in matches]
        if len(matches) == 1:
            evidence = {
                "source": "same_image_roi_match",
                "selection": roi,
                "detection": matches[0][1],
            }
    else:
        raise GoalError("mode must be text or roi")
    if len(candidates) != 1:
        raise GoalError(
            f"Target is not unique: {len(candidates)} matches; supply name/cabinet or tighter ROI"
        )
    target = candidates[0]
    expected = {"push_button": "press", "rotary": "rotate", "toggle": "toggle"}[target["type"]]
    if operation != expected:
        raise GoalError(f"Explicit compatible operation required: {expected}")
    angle = None
    direction = None
    if operation in ("rotate", "toggle"):
        mentioned = re.findall(r"(\d+(?:\.\d+)?)\s*(?:度|°)", request.get("text", ""))
        if len(mentioned) > 1:
            raise GoalError("Multiple angle values are unsupported")
        angle = request.get("angle_deg", float(mentioned[0]) if mentioned else None)
        direction = request.get("joint_direction")
        if (
            angle is None
            or not isinstance(angle, (int, float))
            or not math.isfinite(angle)
            or not 15 <= angle <= 180
        ):
            raise GoalError("Angular goal requires angle_deg in [15,180]")
        if mentioned and float(mentioned[0]) != angle:
            raise GoalError("Text angle conflicts with angle_deg")
        if direction not in (-1, 1) or any(
            s in request.get("text", "") for s in ["顺时针", "逆时针", "向左", "向右"]
        ):
            raise GoalError(
                "Angular goals require calibrated joint_direction; visual direction mapping is not implemented"
            )
    instruction = {"press": "按压", "rotate": "旋转", "toggle": "拨动"}[operation] + target["name"]
    goal = {
        "schema_version": 1,
        "target_id": target["id"],
        "cabinet_id": target["cabinet_id"],
        "target_name": target["name"],
        "target_type": target["type"],
        "operation": operation,
        "desired_result": "press_and_release" if operation == "press" else "specified_angle_change",
        "policy_prompt": instruction + "（" + target["cabinet_id"] + "）",
        "grounding": evidence,
        "localization_status": "reference_image_only" if mode == "roi" else "identity_only",
        "contact_pose_b": None,
    }
    if angle is not None:
        goal.update(angle_deg=angle, joint_direction=direction)
        goal["policy_prompt"] += f"，相对初始位置转动 {angle:g} 度，已标定关节方向 {direction:+d}"
    canonical = json.dumps(
        {
            k: goal.get(k)
            for k in ["target_id", "operation", "desired_result", "angle_deg", "joint_direction"]
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    goal["goal_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return goal


def policy_observation(observation, goal):
    """Attach goal language to live OpenPI input; never expose evaluator joint truth."""
    if goal.get("schema_version") != 1 or not goal.get("goal_id"):
        raise GoalError("Expected resolved goal")
    return {**observation, "prompt": goal["policy_prompt"]}

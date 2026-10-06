"""Streaming developmental evaluator; never fabricates an official verdict.

Uses task-specific scene joint state, NOT robot end pose/contact count.
Must be reset per episode and fed baseline before movement.
"""

import math


class SuccessEvaluator:
    def __init__(self, goal, catalog, criteria):
        self.goal = goal
        target = next((t for t in catalog["targets"] if t["id"] == goal["target_id"]), None)
        if target is None or target["type"] != goal["target_type"]:
            raise ValueError("Goal/catalog mismatch")
        expected = {"push_button": "press", "rotary": "rotate", "toggle": "toggle"}[target["type"]]
        if goal["operation"] != expected:
            raise ValueError("Operation/type mismatch")
        self.joint = target["scene_joint"]
        if criteria.get("schema_version") != 1:
            raise ValueError("Unsupported criteria schema")
        self.cfg = dict(criteria["targets"][goal["target_id"]])
        if expected != "press":
            if not 15 <= goal.get("angle_deg", 0) <= 180 or goal.get("joint_direction") not in (
                -1,
                1,
            ):
                raise ValueError("Explicit angular goal required")
            self.cfg.update(
                threshold=math.radians(goal["angle_deg"]), direction=goal["joint_direction"]
            )
        if self.cfg["unit"] != ("m" if expected == "press" else "rad"):
            raise ValueError("Criterion unit does not match joint type")
        for name in ["threshold", "hold_s", "max_gap_s", "timeout_s"]:
            if not math.isfinite(self.cfg[name]) or self.cfg[name] <= 0:
                raise ValueError("Criteria must be finite and positive")
        if self.cfg["direction"] not in [-1, 1]:
            raise ValueError("Explicit signed direction required")
        if expected == "press" and not 0 <= self.cfg["release_threshold"] < self.cfg["threshold"]:
            raise ValueError("Release hysteresis must be below press threshold")
        self.monitors = criteria.get("non_target_monitors", {})
        if any(not math.isfinite(v) or v <= 0 for v in self.monitors.values()):
            raise ValueError("Monitor thresholds must be positive")
        self.reset()

    def reset(self):
        self.start = self.last = self.baseline = self.above_since = None
        self.operation_observed = self.release_observed = False
        self.invalid = False
        self.timed_out = False
        self.events = []
        self.monitored_baseline = {}
        self.changed = set()
        self.peak = 0.0
        self.samples = 0

    def update(self, sample):
        t = sample["simulation_time_s"]
        if not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError("Finite simulation time required")
        if self.last is not None and t <= self.last:
            raise ValueError("Samples must have strictly increasing timestamps")
        if self.last is not None and t - self.last > self.cfg["max_gap_s"]:
            self.invalid = True
            self.events.append({"type": "data_gap", "time_s": t})
            self.above_since = None
        self.last = t
        self.samples += 1
        positions = sample.get("scene_joint_position", {})
        required = {self.joint, *self.monitors}
        if any(
            k not in positions
            or not isinstance(positions[k], (int, float))
            or not math.isfinite(positions[k])
            for k in required
        ):
            self.invalid = True
            self.above_since = None
            self.events.append({"type": "missing_or_invalid_joint", "time_s": t})
            return self.result()
        if self.start is None:
            self.start = t
            self.baseline = positions[self.joint]
            self.monitored_baseline = {k: positions[k] for k in self.monitors}
        for k, threshold in self.monitors.items():
            if (
                k != self.joint
                and abs(positions[k] - self.monitored_baseline[k]) >= threshold
                and k not in self.changed
            ):
                self.changed.add(k)
                self.events.append({"type": "non_target_state_change", "joint": k, "time_s": t})
        progress = self.cfg["direction"] * (positions[self.joint] - self.baseline)
        self.peak = max(self.peak, progress)
        if t - self.start > self.cfg["timeout_s"] and not (
            self.operation_observed and (self.goal["operation"] != "press" or self.release_observed)
        ):
            self.timed_out = True
        if not self.timed_out:
            if not self.operation_observed:
                if progress >= self.cfg["threshold"]:
                    if self.above_since is None:
                        self.above_since = t
                    if t - self.above_since + 1e-9 >= self.cfg["hold_s"]:
                        self.operation_observed = True
                        self.events.append(
                            {"type": "operation_observed", "time_s": t, "progress": progress}
                        )
                else:
                    self.above_since = None
            elif (
                self.goal["operation"] == "press"
                and not self.release_observed
                and abs(positions[self.joint] - self.baseline) <= self.cfg["release_threshold"]
            ):
                self.release_observed = True
                self.events.append({"type": "release_observed", "time_s": t})
        return self.result()

    def result(self):
        completed = self.operation_observed and (
            self.goal["operation"] != "press" or self.release_observed
        )
        status = (
            "inconclusive"
            if self.invalid or not self.samples
            else "non_target_change"
            if self.changed
            else "timeout"
            if self.timed_out
            else "proxy_pass"
            if completed
            else "in_progress"
        )
        return {
            "schema_version": 1,
            "goal_id": self.goal["goal_id"],
            "target_id": self.goal["target_id"],
            "status": status,
            "proxy_success": bool(
                completed and not self.invalid and not self.changed and not self.timed_out
            ),
            "official_success": None,
            "criterion_source": self.cfg["source"],
            "criterion": dict(self.cfg),
            "operation_observed": self.operation_observed,
            "release_observed": self.release_observed,
            "withdrawal_confirmed": None,
            "collision_attribution": None,
            "peak_signed_change": self.peak,
            "non_target_changed": sorted(self.changed),
            "samples": self.samples,
            "events": list(self.events),
            "limitations": [
                "Development joint-state criterion only; no official referee integration.",
                "Joint changes do not prove collision or causal attribution.",
                "Withdrawal, fall and jitter are not observed by this evaluator.",
            ],
        }

    def finalize(self):
        result = self.result()
        if result["status"] == "in_progress":
            result["status"] = "incomplete"
        return result

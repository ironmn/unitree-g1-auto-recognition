import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from task_goal import GoalError, policy_observation, resolve_goal
from task_success import SuccessEvaluator

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


class GoalTests(unittest.TestCase):
    def setUp(self):
        self.catalog = read("task_catalog.json")
        self.text = read("goal_stop_text.json")
        self.roi = read("goal_stop_roi.json")

    def test_text_and_roi_same_goal(self):
        a = resolve_goal(self.text, self.catalog)
        b = resolve_goal(self.roi, self.catalog)
        self.assertEqual(a["goal_id"], b["goal_id"])
        self.assertEqual(a["target_id"], "cabinet02.stop")
        self.assertIsNone(b["contact_pose_b"])

    def test_color_only_ambiguous(self):
        self.text["text"] = "按压红色按钮"
        with self.assertRaises(GoalError):
            resolve_goal(self.text, self.catalog)

    def test_emergency_is_distinct(self):
        self.text["text"] = "按压急停按钮"
        self.assertEqual(
            resolve_goal(self.text, self.catalog)["target_id"], "cabinet02.emergency_stop"
        )

    def test_negative_instruction_rejected(self):
        self.text["text"] = "不要按压停止按钮"
        with self.assertRaises(GoalError):
            resolve_goal(self.text, self.catalog)

    def test_broad_roi_rejected(self):
        self.roi["bbox_xyxy_norm"] = [0, 0, 1, 1]
        with self.assertRaises(GoalError):
            resolve_goal(self.roi, self.catalog)

    def test_stale_image_rejected(self):
        self.roi["image_id"] = "new_frame"
        with self.assertRaises(GoalError):
            resolve_goal(self.roi, self.catalog)

    def test_wrong_operation_rejected(self):
        self.roi["operation"] = "rotate"
        with self.assertRaises(GoalError):
            resolve_goal(self.roi, self.catalog)

    def test_duplicate_name_requires_cabinet(self):
        duplicate = copy.deepcopy(self.catalog["targets"][0])
        duplicate.update(id="other.stop", cabinet_id="other")
        self.catalog["targets"].append(duplicate)
        self.text.pop("cabinet_id")
        with self.assertRaises(GoalError):
            resolve_goal(self.text, self.catalog)

    def test_policy_input_has_no_joint_truth(self):
        g = resolve_goal(self.text, self.catalog)
        self.assertEqual(set(policy_observation({"state": []}, g)), {"state", "prompt"})

    def test_angle_and_direction_are_part_of_goal(self):
        request = {
            "schema_version": 1,
            "mode": "text",
            "text": "旋转旋钮开关90度",
            "joint_direction": 1,
        }
        g = resolve_goal(request, self.catalog)
        self.assertEqual(g["angle_deg"], 90)
        ev = SuccessEvaluator(g, self.catalog, read("task_success_dev.json"))
        self.assertAlmostEqual(ev.cfg["threshold"], 1.57079632679)
        request.pop("joint_direction")
        with self.assertRaises(GoalError):
            resolve_goal(request, self.catalog)


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.catalog = read("task_catalog.json")
        self.goal = resolve_goal(read("goal_stop_text.json"), self.catalog)
        self.criteria = read("task_success_dev.json")
        self.e = SuccessEvaluator(self.goal, self.catalog, self.criteria)

    def sample(self, t, value=0, other=None):
        joints = {k: 0.0 for k in self.criteria["non_target_monitors"]}
        joints[self.e.joint] = value
        joints.update(other or {})
        return {"simulation_time_s": t, "scene_joint_position": joints}

    def press(self):
        for t, v in [(0, 0), (0.04, 0.002), (0.08, 0.002), (0.12, 0.002), (0.16, 0)]:
            self.e.update(self.sample(t, v))

    def test_press_release_once_and_official_unknown(self):
        self.press()
        self.e.update(self.sample(0.20, 0.002))
        r = self.e.finalize()
        self.assertTrue(r["proxy_success"])
        self.assertIsNone(r["official_success"])
        self.assertEqual(sum(x["type"] == "operation_observed" for x in r["events"]), 1)

    def test_spike_does_not_pass(self):
        for t, v in [(0, 0), (0.04, 0.002), (0.08, 0)]:
            self.e.update(self.sample(t, v))
        self.assertFalse(self.e.finalize()["proxy_success"])

    def test_wrong_direction_does_not_pass(self):
        for t in [0, 0.04, 0.08, 0.12]:
            self.e.update(self.sample(t, 0 if t == 0 else -0.003))
        self.assertFalse(self.e.finalize()["operation_observed"])

    def test_no_release_incomplete(self):
        for t, v in [(0, 0), (0.04, 0.002), (0.08, 0.002), (0.12, 0.002)]:
            self.e.update(self.sample(t, v))
        self.assertEqual(self.e.finalize()["status"], "incomplete")

    def test_other_target_changed(self):
        self.press()
        self.e.update(
            self.sample(0.20, 0, {"Group_Interactive_DistributionBox_02_halt_button_joint": 0.003})
        )
        self.assertEqual(self.e.finalize()["status"], "non_target_change")

    def test_missing_joint_unknown(self):
        self.e.update({"simulation_time_s": 0, "scene_joint_position": {}})
        self.press_after_missing()
        self.assertEqual(self.e.finalize()["status"], "inconclusive")

    def press_after_missing(self):
        for t, v in [(0.04, 0), (0.08, 0.002), (0.12, 0.002), (0.16, 0.002), (0.20, 0)]:
            self.e.update(self.sample(t, v))

    def test_gap_invalidates_pass(self):
        self.press()
        self.e.update(self.sample(1, 0))
        self.assertEqual(self.e.finalize()["status"], "inconclusive")

    def test_reordered_time_rejected(self):
        self.e.update(self.sample(0))
        with self.assertRaises(ValueError):
            self.e.update(self.sample(0))

    def test_reset_clears_latched_event(self):
        self.press()
        self.e.reset()
        self.assertFalse(self.e.finalize()["operation_observed"])

    def test_timeout_before_completion(self):
        self.e.cfg = copy.deepcopy(self.e.cfg)
        self.e.cfg["timeout_s"] = 0.1
        for t in [0, 0.04, 0.08, 0.12]:
            self.e.update(self.sample(t))
        self.assertEqual(self.e.finalize()["status"], "timeout")


if __name__ == "__main__":
    unittest.main()

"""Exercise the pose/claw boundary and binary transport without a robot or GPU."""

import io
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_policy_client import (  # noqa: E402
    IMAGE_KEYS,
    ActionLimits,
    PolicyClient,
    guarded_right_action,
    quaternion_angle,
    right_gripper_motor,
)


def pose():
    value = np.zeros(18, dtype=np.float32)
    value[6] = value[13] = 1
    value[14:] = 0.5
    return value


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TestPolicyClient(unittest.TestCase):
    def test_transport_preserves_rgb_state_prompt_seed_and_absolute_output(self):
        images = {key: np.full((4, 6, 3), [17, 41, 139], np.uint8) for key in IMAGE_KEYS}
        predicted = np.tile(pose(), (24, 1))
        predicted[:, 7] = 0.3
        payload = io.BytesIO()
        np.savez_compressed(payload, actions=predicted, sampling_ms=np.float32(45))

        def open_request(request, timeout):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.full_url, "http://127.0.0.1:8766/infer")
            with np.load(io.BytesIO(request.data), allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["state"], pose())
                self.assertEqual(str(archive["prompt"]), "按压停止按钮")
                self.assertEqual(int(archive["seed"]), 2**32 - 1)
                self.assertEqual(archive["seed"].dtype, np.uint32)
                np.testing.assert_array_equal(archive[IMAGE_KEYS[0]], images[IMAGE_KEYS[0]])
            return Response(payload.getvalue())

        with patch("urllib.request.urlopen", side_effect=open_request):
            result, timing = PolicyClient().infer(pose(), "按压停止按钮", images, 2**32 - 1)
        np.testing.assert_array_equal(result, predicted)
        self.assertEqual(timing, 45)

    def test_bad_response_never_becomes_a_command(self):
        images = {key: np.zeros((4, 6, 3), np.uint8) for key in IMAGE_KEYS}
        for actions, timing in [
            (np.zeros((24, 32)), 1),
            (np.full((24, 18), np.nan), 1),
            (np.zeros((24, 18)), -1),
        ]:
            with self.subTest(shape=actions.shape, timing=timing):
                payload = io.BytesIO()
                np.savez_compressed(payload, actions=actions, sampling_ms=np.asarray(timing))
                with patch("urllib.request.urlopen", return_value=Response(payload.getvalue())):
                    with self.assertRaises(ValueError):
                        PolicyClient().infer(pose(), "press", images, 1)

    def test_invalid_seed_and_image_fail_before_network(self):
        images = {key: np.zeros((4, 6, 3), np.uint8) for key in IMAGE_KEYS}
        with patch("urllib.request.urlopen") as network:
            for seed in (-1, 2**32, True):
                with self.assertRaises(ValueError):
                    PolicyClient().infer(pose(), "press", images, seed)
            images[IMAGE_KEYS[0]] = np.zeros((4, 6, 3), np.float32)
            with self.assertRaises(ValueError):
                PolicyClient().infer(pose(), "press", images, 1)
            network.assert_not_called()


class TestActionBoundary(unittest.TestCase):
    def test_absolute_base_pose_and_gripper_units(self):
        current = pose()
        current[7:10] = [0.3, -0.1, 0.2]
        raw = current.copy()
        raw[7:10] += [0.005, 0.002, 0.001]
        raw[16:] = [0, 1]
        command, constraints = guarded_right_action(raw, current)
        # Keeping absolute base position catches accidental delta/motor conversion.
        np.testing.assert_allclose(command[7:10], [0.305, -0.098, 0.201])
        np.testing.assert_allclose(right_gripper_motor(command), [-1, 2])
        self.assertFalse(constraints["position_limited"])

    def test_shortest_quaternion_sign_and_rotation_limit(self):
        current = pose()
        raw = pose()
        raw[13] = -2
        command, details = guarded_right_action(raw, current)
        self.assertAlmostEqual(quaternion_angle(command[10:14], current[10:14]), 0)
        self.assertTrue(details["quaternion_normalized"])
        raw[10:14] = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        command, details = guarded_right_action(raw, current, ActionLimits(orientation_step_deg=10))
        self.assertAlmostEqual(
            math.degrees(quaternion_angle(command[10:14], current[10:14])), 10, places=4
        )
        self.assertTrue(details["orientation_limited"])

    def test_projection_is_logged_and_left_actions_are_not_executed(self):
        current = pose()
        raw = pose()
        raw[:3] = [9, 8, 7]
        raw[7:10] = [0.1, 0, 0]
        raw[16:] = [-0.5, 1.2]
        command, details = guarded_right_action(raw, current)
        np.testing.assert_array_equal(command[:7], current[:7])
        np.testing.assert_allclose(command[7:10], [0.02, 0, 0])
        np.testing.assert_array_equal(command[16:], [0, 1])
        self.assertTrue(details["position_limited"])
        self.assertTrue(details["gripper_clipped"])
        self.assertAlmostEqual(details["position_projection_m"], 0.08, places=6)

    def test_gross_jump_and_zero_quaternion_reject(self):
        current = pose()
        raw = pose()
        raw[7] = 0.6
        with self.assertRaisesRegex(ValueError, "pose-jump"):
            guarded_right_action(raw, current)
        raw[7] = 0
        raw[10:14] = 0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            guarded_right_action(raw, current)


if __name__ == "__main__":
    unittest.main()

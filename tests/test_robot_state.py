import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location(
    "robot_state", Path(__file__).parents[1] / "scripts/robot_state.py"
)
state = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = state
spec.loader.exec_module(state)


def names():
    return ["g1_pick_" + n for n in state.BODY] + [
        f"g1_pick_idx{i}_gripper_{side}_{n}"
        for side in ("l", "r")
        for i, n in enumerate(state.FINGERS)
    ]


class StateTests(unittest.TestCase):
    def reader(self, profile="full"):
        self.q = np.arange(67, dtype=float)
        self.v = -np.arange(65, dtype=float)
        joints = [
            state.Joint(n, 3, i + 22, i + 20, True, [-3.0, 3.0]) for i, n in enumerate(names())
        ]
        joints += [
            state.Joint("panel_button", 2, 0, 0, False, [0.0, 0.0]),
            state.Joint("g1_pick_floating_base_joint", 0, 15, 14, False, [0.0, 0.0]),
        ]
        return state.StateReader(
            list(reversed(joints)), lambda: (self.q, self.v, 1.2), profile=profile
        )

    def test_order_offsets_units_and_profiles(self):
        for profile, size in [("full", 45), ("arms", 30), ("right", 15)]:
            r = self.reader(profile)
            row = r.read(sequence=5, simulate_index=42)
            self.assertEqual(len(row["state"]), size)
            self.assertEqual(row["sequence"], 5)
            self.assertEqual(row["simulate_index"], 42)
            self.assertEqual(row["state"], [self.q[j.qpos_address] for j in r.joints])
            self.assertEqual(row["joint_velocity"], [self.v[j.qvel_address] for j in r.joints])
            self.assertEqual(len(row["groups"]["gripper_right"]["position"]), 8)
            self.assertEqual(r.manifest["state_units"], ["rad"] * size)
            self.assertFalse(any("floating" in n for n in r.manifest["state_names"]))

    def test_invalid_feedback_rejected(self):
        r = self.reader()
        self.q[22] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            r.read()
        r.sample = lambda: (np.zeros(7), np.zeros(6), None)
        with self.assertRaisesRegex(ValueError, "does not match"):
            r.read()

    def test_missing_and_ambiguous_joint_rejected(self):
        r = self.reader()
        for joints in [r.joints[1:], r.joints + [r.joints[0]]]:
            with self.assertRaisesRegex(ValueError, "Expected one"):
                state.StateReader(joints, r.sample)

    def test_local_adapter_reads_current_physics_data(self):
        import mujoco

        xml = (
            "<mujoco><worldbody>"
            + "".join(
                f'<body><joint name="{n}"/><geom type="sphere" size="0.01" mass="0.1"/></body>'
                for n in names()
            )
            + "</worldbody></mujoco>"
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        env = SimpleNamespace(gym=SimpleNamespace(_mjModel=model, _mjData=data))
        reader = state.StateReader.from_env(env)
        data.qpos[:] = np.arange(45) / 100
        data.qvel[:] = np.arange(45) / 10
        data.time = 2.5
        row = reader.read()
        self.assertEqual(row["state"], data.qpos.tolist())
        self.assertEqual(row["joint_velocity"], data.qvel.tolist())
        self.assertEqual(row["simulation_time_s"], 2.5)
        data.qpos[0] = 0.8
        self.assertEqual(reader.read()["state"][0], 0.8)


if __name__ == "__main__":
    unittest.main()

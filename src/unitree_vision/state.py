"""G1 joint/gripper observations. CLI reads server state without changing controls.

For a controller using OrcaGymLocalEnv, use StateReader.from_env(env).read()
in that controller's own stepping thread: server feedback may lag local physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
ARM = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
]
BODY = (
    [f"{side}_{part}_joint" for side in ("left", "right") for part in LEG]
    + [f"waist_{part}_joint" for part in ("yaw", "roll", "pitch")]
    + [f"{side}_{part}_joint" for side in ("left", "right") for part in ARM]
)
# Explicit linkage order; the idx prefix is resolved from the loaded model.
FINGERS = [
    "inner_joint1",
    "inner_joint3",
    "inner_joint4",
    "inner_joint2",
    "outer_joint1",
    "outer_joint3",
    "outer_joint4",
    "outer_joint2",
]


@dataclass(frozen=True)
class Joint:
    name: str
    kind: int
    qpos_address: int
    qvel_address: int
    limited: bool
    limits: list[float]


class StateReader:
    """Fixed ordering with strict names, dimensions and finite-value validation."""

    def __init__(self, joints, sample, *, agent="g1_pick", profile="full", source="local"):
        if profile not in ("full", "arms", "right"):
            raise ValueError("profile must be full, arms or right")
        prefix = agent + "_"
        robot = [j for j in joints if j.name.startswith(prefix)]

        def resolve(suffix):
            matches = [
                j for j in robot if j.name == prefix + suffix or j.name.endswith("_" + suffix)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {agent}/{suffix}; found {[j.name for j in matches]}"
                )
            j = matches[0]
            if j.kind not in (2, 3):
                raise ValueError(f"{j.name}: expected scalar hinge/slide joint, got {j.kind}")
            return j

        body_suffixes = (
            BODY
            if profile == "full"
            else [
                f"{side}_{part}_joint"
                for side in (("left", "right") if profile == "arms" else ("right",))
                for part in ARM
            ]
        )
        self.groups = {"body": [resolve(n) for n in body_suffixes]}
        for side, token in [("left", "l"), ("right", "r")]:
            if profile == "right" and side == "left":
                continue
            self.groups[f"gripper_{side}"] = [resolve(f"gripper_{token}_{n}") for n in FINGERS]
        self.joints = [j for group in self.groups.values() for j in group]
        self.sample = sample
        self.source = source
        manifest = {
            "schema_version": 1,
            "agent": agent,
            "profile": profile,
            "source": source,
            "state_names": [j.name for j in self.joints],
            "state_units": ["rad" if j.kind == 3 else "m" for j in self.joints],
            "velocity_units": ["rad/s" if j.kind == 3 else "m/s" for j in self.joints],
            "groups": {k: [j.name for j in v] for k, v in self.groups.items()},
            "joints": [asdict(j) for j in self.joints],
            "state_definition": "Measured qpos in state_names order; velocities stored separately.",
            "gripper_definition": "Eight measured linkage angles per hand, not aperture or command.",
            "normalization": None,
        }
        self.schema_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        self.manifest = dict(manifest, schema_id=self.schema_id)

    @classmethod
    def from_env(cls, env, *, agent="g1_pick", profile="full"):
        """Read the controller's actual MuJoCo data, without creating/resetting an env.

        Uses the installed OrcaGym 26.8.2 local MuJoCo bridge. Invoke after physics
        step and before render, on the same thread; no locks or stepping are added.
        """
        import mujoco

        model, data = env.gym._mjModel, env.gym._mjData
        joints = [
            Joint(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "",
                int(model.jnt_type[i]),
                int(model.jnt_qposadr[i]),
                int(model.jnt_dofadr[i]),
                bool(model.jnt_limited[i]),
                model.jnt_range[i].tolist(),
            )
            for i in range(model.njnt)
        ]
        return cls(
            joints,
            lambda: (data.qpos.copy(), data.qvel.copy(), float(data.time)),
            agent=agent,
            profile=profile,
            source="controller_local_mujoco",
        )

    @classmethod
    def from_server(cls, stub, *, agent="g1_pick", profile="full", timeout=5):
        from orca_gym.protos import mjc_message_pb2 as pb

        response = stub.QueryAllJoints(pb.QueryAllJointsRequest(), timeout=timeout)
        joints = [
            Joint(
                j.name, j.type, j.qpos_idx_start, j.qvel_idx_start, bool(j.limited), list(j.range)
            )
            for j in response.joint_info
        ]

        def sample():
            r = stub.QueryAllQposQvelQacc(pb.QueryAllQposQvelQaccRequest(), timeout=timeout)
            return np.asarray(r.qpos), np.asarray(r.qvel), None

        return cls(joints, sample, agent=agent, profile=profile, source="server_grpc")

    def read(self, *, sequence=0, simulate_index=None):
        started = time.monotonic_ns()
        qpos, qvel, simulation_time = self.sample()
        qa = [j.qpos_address for j in self.joints]
        va = [j.qvel_address for j in self.joints]
        if min(qa + va) < 0 or max(qa) >= len(qpos) or max(va) >= len(qvel):
            raise ValueError(
                "State array does not match the loaded robot model; start the correct Runtime."
            )
        if np.asarray(qpos).ndim != 1 or np.asarray(qvel).ndim != 1:
            raise ValueError("State arrays must be one dimensional")
        if simulation_time is not None and not np.isfinite(simulation_time):
            raise ValueError("Non-finite simulation time")
        position, velocity = np.asarray(qpos)[qa], np.asarray(qvel)[va]
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("Non-finite joint feedback; observation rejected")
        groups, offset = {}, 0
        for name, joints in self.groups.items():
            end = offset + len(joints)
            groups[name] = {
                "position": position[offset:end].tolist(),
                "velocity": velocity[offset:end].tolist(),
            }
            offset = end
        return {
            "schema_id": self.schema_id,
            "sequence": sequence,
            "wall_time_ns": time.time_ns(),
            "read_started_monotonic_ns": started,
            "read_finished_monotonic_ns": time.monotonic_ns(),
            "simulation_time_s": simulation_time,
            "simulate_index": simulate_index,
            "state": position.tolist(),
            "joint_velocity": velocity.tolist(),
            "groups": groups,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grpc-address", default="localhost:50051")
    p.add_argument("--agent", default="g1_pick")
    p.add_argument("--profile", choices=["full", "arms", "right"], default="full")
    p.add_argument("--hz", type=float, default=20)
    p.add_argument("--duration", type=float, default=10, help="Seconds; 0 runs until Ctrl+C")
    p.add_argument("--timeout", type=float, default=5)
    p.add_argument("--output", type=Path, default=Path("data/robot_state"))
    args = p.parse_args()
    if not (
        np.isfinite(args.hz)
        and 0 < args.hz <= 200
        and np.isfinite(args.duration)
        and args.duration >= 0
        and np.isfinite(args.timeout)
        and args.timeout > 0
    ):
        p.error("hz must be (0, 200], duration >= 0 and timeout > 0; all finite")
    import grpc
    from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

    try:
        with grpc.insecure_channel(args.grpc_address) as channel:
            grpc.channel_ready_future(channel).result(timeout=args.timeout)
            reader = StateReader.from_server(
                GrpcServiceStub(channel),
                agent=args.agent,
                profile=args.profile,
                timeout=args.timeout,
            )
            first = reader.read()
            folder = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            folder.mkdir(parents=True)
            (folder / "manifest.json").write_text(
                json.dumps(reader.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"State dimension={len(first['state'])}; source=server_grpc; output={folder.resolve()}"
            )
            print("Server diagnostic stream; for local controllers use StateReader.from_env(env).")
            count = 0
            start = time.monotonic()
            with (folder / "states.jsonl").open("w", encoding="utf-8") as f:
                while True:
                    row = first if count == 0 else reader.read(sequence=count)
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    f.flush()
                    count += 1
                    if count == 1 or count % max(1, round(args.hz)) == 0:
                        print(
                            f"{count} samples; gripper R rad: {np.round(row['groups']['gripper_right']['position'], 4).tolist()}"
                        )
                    deadline = start + count / args.hz
                    if args.duration and deadline - start >= args.duration:
                        break
                    time.sleep(max(0, deadline - time.monotonic()))
            print(f"Saved {count} observations.")
    except KeyboardInterrupt:
        print("Stopped; written observations preserved.")
    except Exception as exc:
        p.exit(
            1, f"State collection failed: {exc}\nCheck OrcaLab Runtime, g1_pick layout and port.\n"
        )


if __name__ == "__main__":
    main()

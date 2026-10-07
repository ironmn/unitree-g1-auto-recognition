"""Run a WSL G1 model through the official Windows OrcaLab right-arm OSC.

This is a simulator evaluation command, not a unittest. Only --execute creates
or resets the SDK environment. Frames, raw predictions, constrained commands and
scene-state proxy verdicts are preserved; no waypoint demonstration is replayed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from g1_policy_client import (
    IMAGE_KEYS,
    ActionLimits,
    PolicyClient,
    guarded_right_action,
    right_gripper_motor,
)
from PIL import Image
from task_goal import resolve_goal
from task_success import SuccessEvaluator

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "press": ("按压停止按钮", "goal_stop_text.json"),
    "rotate": ("旋转旋钮", "goal_rotate_demo.json"),
    "toggle": ("拨动拨杆式按钮", "goal_toggle_demo.json"),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def rgb_model_image(bgr):
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    image = Image.fromarray(rgb)
    scale = 224 / max(image.size)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (224, 224))
    canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
    return np.asarray(canvas)


class OfficialSession:
    """One owner for SDK reset, controls, state, rendering and camera receivers."""

    def __init__(self, args):
        self.args, self.env, self.strip = args, None, None
        self.streams, self.started_cameras = {}, []
        self.index = 0

    def open(self):
        folder = self.args.official_root / "src/examples/dataCollection/unitree_g1"
        for path in (
            self.args.official_root / "src",
            folder,
            folder.parent / "common",
            ROOT / "src",
        ):
            sys.path.insert(0, str(path))
        import mj_joint_strip
        from conf import g1_pick_osc_conf as conf
        from controllers.controller_2f85_reverse import Controller2F85Reverse
        from controllers.controllers import (
            create_arm_osc_controller,
            create_gripper_2f85_reverse_controller,
            install_osc_patches,
        )
        from dataCollectionManager.data_collection_manager import DataCollectionManager

        from unitree_vision.camera import check_response, resolve_camera
        from unitree_vision.streams import IndexedCameraStream

        self.conf = conf
        install_osc_patches(dls_lambda=0.23, dls_sigma_th=0.12, null_kp=10)
        keep = mj_joint_strip.KEEP_DEFAULT + tuple(conf.l_arm["joint_names"])
        self.strip = mj_joint_strip.install(
            None,
            self.args.agent,
            keep=keep,
            kill_collision=True,
            required_cameras=("camera_head_color", "camera_wrist_r_color"),
        )

        # Gym deep-copies registered kwargs. A bound method would deepcopy this
        # session (including SDK modules/threads); plain functions retain identity.
        def observation_callback(env):
            return self.observation(env)

        # Assets are already loaded by the GUI. No scene spawning/randomization.
        self.manager = DataCollectionManager(
            agent_name=self.args.agent,
            env_name="G1ModelEvaluation",
            entry_point="envs.dataCollection.dataCollection_env:DataCollectionEnv",
            default_joint_values={},
            obs_callback=observation_callback,
            scene_manager=None,
            frame_skip=5,
            time_step=0.001,
            orcagym_addr=self.args.grpc_address,
        )
        self.env = env = self.manager.env
        if not self.strip.applied:
            raise RuntimeError("Training-compatible stripped model was not applied")
        mj_joint_strip.finish_install(env, self.strip, self.args.agent)
        env.set_sync_render(True)
        env.set_render_fps(200)
        actuator_names = [env.actuator(n) for n in conf.gripper_r["actuator_names"]]
        self.gripper = create_gripper_2f85_reverse_controller(
            env,
            conf.gripper_r,
            conf.base_body,
            actuator_names,
            dict(zip(actuator_names, conf.gripper_r["init_ctrl"])),
            Controller2F85Reverse.ControllerType.DATA,
        )
        self.gripper.update_ctrl(np.full(2, -1, dtype=np.float32))
        self.manager.add_controller(self.gripper)
        actuator_names = [env.actuator(n) for n in conf.r_arm["motors_names"]]
        self.arm = create_arm_osc_controller(
            env,
            conf.r_arm,
            conf.base_body,
            actuator_names,
            dict(zip(actuator_names, conf.r_arm["motors_init_ctrl"])),
        )
        self.manager.add_controller(self.arm)
        import grpc
        from orca_gym.protos import mjc_message_pb2 as pb
        from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

        self.channel = grpc.insecure_channel(self.args.grpc_address)
        self.stub = GrpcServiceStub(self.channel)
        names = list(
            check_response(
                self.stub.GetCameraNames(pb.GetCameraNamesRequest(), timeout=5), "GetCameraNames"
            ).camera_names
        )
        self.camera_info = {}
        for role in ("head", "wrist_r"):
            name = resolve_camera(names, role)
            properties = check_response(
                self.stub.GetCameraProperties(
                    pb.GetCameraPropertiesRequest(camera_name=name), timeout=5
                ),
                name,
            )
            if not properties.capture_rgb or not 0 < properties.color_port < 65536:
                raise RuntimeError("Enable RGB and camera color port in the loaded layout")
            if not properties.streaming_enabled:
                check_response(
                    self.stub.SetStreamingEnabled(
                        pb.SetStreamingEnabledRequest(camera_name=name, enabled=True), timeout=5
                    ),
                    name,
                )
                self.started_cameras.append(name)
            self.camera_info[role] = {
                "name": name,
                "width": properties.width,
                "height": properties.height,
                "port": properties.color_port,
            }
            stream = IndexedCameraStream(role, "127.0.0.1", properties.color_port, capacity=16)
            self.streams[role] = stream
            stream.thread.start()

    def observation(self, env):
        if env.model.nu == 0:
            return {"state": np.zeros(18, dtype=np.float32)}
        conf = self.conf
        sites = [env.site(conf.l_arm["ee_site_name"]), env.site(conf.r_arm["ee_site_name"])]
        poses = env.query_site_pos_and_quat_B(sites, [env.body(conf.base_body)])
        actuator_dict = getattr(env.model, "_actuator_dict", {}) or {}
        grippers = []
        for configuration in (conf.gripper_l, conf.gripper_r):
            for name, bounds in zip(
                configuration["actuator_names"], configuration["actuator_ranges"]
            ):
                full = env.actuator(name)
                motor = (
                    float(env.ctrl[env.model.actuator_name2id(full)])
                    if full in actuator_dict
                    else 0.0
                )
                grippers.append(float(np.clip((motor - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)))
        state = np.concatenate(
            [
                np.asarray(poses[sites[0]]["xpos"]).reshape(3),
                np.asarray(poses[sites[0]]["xquat"])[[1, 2, 3, 0]],
                np.asarray(poses[sites[1]]["xpos"]).reshape(3),
                np.asarray(poses[sites[1]]["xquat"])[[1, 2, 3, 0]],
                np.asarray(grippers),
            ]
        ).astype(np.float32)
        if not np.isfinite(state).all():
            raise RuntimeError("Non-finite observation state")
        return {"state": state}

    def reset(self):
        env, conf = self.env, self.conf
        env.reset()
        defaults = dict(zip(conf.l_arm["joint_names"], [0, 0.127, 0, 1.5708, 0, 0, 0]))
        defaults.update(dict.fromkeys(conf.r_arm["joint_names"], 0))
        env.set_default_joint_values(defaults)
        env.mj_forward()
        self.manager.set_init_ctrl()
        for controller in self.manager.controllers:
            controller.reset()
        state = self.observation(env)["state"]
        self.arm.update_action_position(state[7:10])
        self.arm.update_action_axisangle(state[10:14])
        self.gripper.update_ctrl(np.full(2, -1, dtype=np.float32))
        ctrl = self.manager.run_controllers()
        env.ctrl = ctrl
        env.set_ctrl(ctrl)
        env.mj_forward()
        for stream in self.streams.values():
            stream.drain()
        return self.observation(env)["state"]

    def paired_images(self):
        index = self.index
        self.index += 1
        state = self.observation(self.env)["state"]
        for stream in self.streams.values():
            stream.drain()
        deadline = time.monotonic() + self.args.camera_timeout
        matched = {}
        rendered = False
        retry_after = 0.0
        while time.monotonic() < deadline:
            if not rendered or time.monotonic() >= retry_after:
                # Same physics pose/index is rerendered only to recover missing IDR.
                self.env.render(simulate_index=index, request_idr=True)
                rendered, retry_after = True, time.monotonic() + 1
            for role, stream in self.streams.items():
                for frame, metadata in stream.drain():
                    if metadata["simulate_index"] == index and role not in matched:
                        matched[role] = (frame, metadata)
            if len(matched) == 2:
                images = {
                    key: rgb_model_image(matched[role][0])
                    for key, role in zip(IMAGE_KEYS, ("head", "wrist_r"))
                }
                return state, images, index, {role: matched[role][1] for role in matched}
            time.sleep(0.005)
        raise TimeoutError("No exactly aligned head/right-wrist camera pair")

    def command(self, action):
        self.arm.update_action_position(action[7:10])
        self.arm.update_action_axisangle(action[10:14])
        self.gripper.update_ctrl(
            right_gripper_motor(action, self.conf.gripper_r["actuator_ranges"])
        )

    def step_until(self, target_time):
        while self.simulation_time < target_time - 1e-9:
            self.env.step(self.manager.run_controllers())
            data = self.env.gym._mjData
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Non-finite simulator physics")

    @property
    def simulation_time(self):
        return float(self.env.gym._mjData.time)

    def scene_sample(self):
        import mujoco

        model, data = self.env.gym._mjModel, self.env.gym._mjData
        positions = {}
        for i in range(model.njnt):
            if int(model.jnt_type[i]) not in (2, 3):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            if not name.startswith(self.args.agent + "_"):
                positions[name] = float(data.qpos[model.jnt_qposadr[i]])
        return {
            "simulation_time_s": self.simulation_time,
            "scene_joint_position": positions,
            "contact_count": int(data.ncon),
        }

    def close(self):
        for stream in self.streams.values():
            stream.stop()
        if hasattr(self, "stub"):
            from orca_gym.protos import mjc_message_pb2 as pb

            for name in self.started_cameras:
                try:
                    self.stub.SetStreamingEnabled(
                        pb.SetStreamingEnabledRequest(camera_name=name, enabled=False), timeout=5
                    )
                except Exception as exc:
                    print("Stream cleanup:", exc, flush=True)
            self.channel.close()
        if self.env is not None:
            self.env.close()
        if self.strip is not None:
            self.strip.restore()


def run_trial(session, client, task, repeat, args, limits, directory):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "frames").mkdir()
    prompt, request_file = TASKS[task]
    catalog = read_json(ROOT / "configs/task_catalog.json")
    goal = resolve_goal(read_json(ROOT / "configs" / request_file), catalog)
    evaluator = SuccessEvaluator(goal, catalog, read_json(ROOT / "configs/task_success_dev.json"))
    seed = (args.seed + list(TASKS).index(task) * 1000 + repeat) % 2**32
    record = {
        "task": task,
        "repeat": repeat,
        "seed": seed,
        "prompt": prompt,
        "goal": goal,
        "status": "starting",
    }
    sampling_ms, request_ms = [], []
    counts = {
        name: 0
        for name in (
            "position_limited",
            "orientation_limited",
            "gripper_clipped",
            "quaternion_normalized",
        )
    }
    frame_count = chunk_count = 0
    wall_start = time.monotonic()
    start_simulation_time = None
    stop_reason = "time_limit"
    try:
        previous = session.reset()
        start_simulation_time = session.simulation_time
        initial_sample = session.scene_sample()
        record.update(initial_state=previous.tolist(), initial_scene_sample=initial_sample)
        baseline_verdict = evaluator.update(initial_sample)  # BEFORE first learned motion.
        if baseline_verdict["status"] == "inconclusive":
            raise RuntimeError("Scene is missing required target/non-target joint-state baseline")
        with (directory / "trajectory.jsonl").open("w", encoding="utf-8") as trace:
            while frame_count < math.ceil(args.seconds * args.fps):
                if time.monotonic() - wall_start > args.wall_timeout:
                    raise TimeoutError("Trial wall-time limit")
                state, images, index, metadata = session.paired_images()
                for key, role in zip(IMAGE_KEYS, ("head", "wrist_r")):
                    Image.fromarray(images[key]).save(
                        directory / "frames" / f"{chunk_count:05d}_{role}.png"
                    )
                t0 = time.monotonic()
                actions, timing = client.infer(state, prompt, images, (seed + chunk_count) % 2**32)
                request_ms.append((time.monotonic() - t0) * 1000)
                sampling_ms.append(timing)
                np.savez_compressed(
                    directory / f"prediction_{chunk_count:05d}.npz",
                    state=state,
                    actions=actions,
                    simulate_index=index,
                    seed=np.uint32((seed + chunk_count) % 2**32),
                )
                # First command of each replan is bounded against actual feedback.
                previous = state
                for k in range(min(args.execute_horizon, len(actions))):
                    if frame_count >= math.ceil(args.seconds * args.fps):
                        break
                    command, constraints = guarded_right_action(actions[k], previous, limits)
                    for name in counts:
                        counts[name] += int(constraints[name])
                    session.command(command)
                    session.step_until(start_simulation_time + (frame_count + 1) / args.fps)
                    frame_count += 1
                    measured = session.observation(session.env)["state"]
                    sample = session.scene_sample()
                    result = evaluator.update(sample)
                    sample.update(
                        {
                            "frame": frame_count,
                            "chunk": chunk_count,
                            "action_index": k,
                            "observation_index": index,
                            "state": measured.tolist(),
                            "raw_action": actions[k].tolist(),
                            "command": command.tolist(),
                            "constraints": constraints,
                            "camera_metadata": metadata if k == 0 else None,
                            "sampling_ms": timing if k == 0 else None,
                            "right_tracking_error_m": float(
                                np.linalg.norm(measured[7:10] - command[7:10])
                            ),
                        }
                    )
                    trace.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    trace.flush()
                    previous = command
                    if result["status"] in (
                        "proxy_pass",
                        "non_target_change",
                        "inconclusive",
                        "timeout",
                    ):
                        stop_reason = result["status"]
                        break
                chunk_count += 1
                # Render the resulting physics state for the GUI between replans.
                print(
                    f"{task} trial {repeat} frame={frame_count} sim={session.simulation_time - start_simulation_time:.3f}s status={evaluator.result()['status']}",
                    flush=True,
                )
                if stop_reason != "time_limit":
                    break
        record["status"] = "completed"
    except Exception as exc:
        stop_reason = "execution_error"
        record.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
    finally:
        record.update(
            stop_reason=stop_reason,
            frames_executed=frame_count,
            model_requests=len(sampling_ms),
            simulation_duration_s=session.simulation_time - start_simulation_time
            if start_simulation_time is not None
            else 0,
            wall_duration_s=time.monotonic() - wall_start,
            sampling_ms_mean=float(np.mean(sampling_ms)) if sampling_ms else None,
            request_ms_mean=float(np.mean(request_ms)) if request_ms else None,
            constraint_counts=counts,
            evaluation=evaluator.finalize(),
            policy_is_constrained=True,
            official_success=None,
        )
        record["trial_proxy_success"] = bool(
            record["status"] == "completed" and record["evaluation"]["proxy_success"]
        )
        write_json(directory / "result.json", record)
    return record


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create/reset the SDK scene and execute learned actions",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--grpc-address", default="localhost:50051")
    parser.add_argument("--agent", default="g1_pick")
    parser.add_argument("--official-root", type=Path, default=ROOT.parent / "Binjiang_Competition")
    parser.add_argument("--output", type=Path, default=ROOT / "data/orcalab_model_eval_20261007")
    parser.add_argument("--tasks", default="press,rotate,toggle")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=int, choices=[24], default=24)
    parser.add_argument("--execute-horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--camera-timeout", type=float, default=15)
    parser.add_argument("--wall-timeout", type=float, default=600)
    parser.add_argument("--position-step-m", type=float, default=0.02)
    parser.add_argument("--orientation-step-deg", type=float, default=10)
    parser.add_argument("--reject-position-jump-m", type=float, default=0.5)
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in TASKS for task in tasks):
        parser.error("--tasks must be unique names from press,rotate,toggle")
    if not 1 <= args.execute_horizon <= 24 or args.repeats < 1 or not 0 < args.seconds <= 180:
        parser.error("Invalid trial size")
    if args.camera_timeout <= 0 or args.wall_timeout <= 0:
        parser.error("Timeouts must be positive")
    limits = ActionLimits(
        args.position_step_m, args.orientation_step_deg, args.reject_position_jump_m
    )
    client = PolicyClient(args.url)
    health = client.health()
    if health.get("ready") is not True or any(
        health.get(key) != value
        for key, value in {"horizon": 24, "action_dim": 18, "fps": 24}.items()
    ):
        raise ValueError("Model service health does not match the G1 evaluation contract")
    host, port = args.grpc_address.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=5):
        pass
    print(
        json.dumps(
            {"policy_health": health, "runtime_tcp_connected": True, "execute": args.execute},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not args.execute:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    session = OfficialSession(args)
    records = []
    report = {
        "schema_version": 1,
        "policy_health": health,
        "configuration": {
            **vars(args),
            "official_root": str(args.official_root),
            "output": str(args.output),
        },
        "action_limits": dataclasses.asdict(limits),
        "controller": "official right-arm base-frame absolute pose OSC; no learned left-arm control",
        "action_schema": "18D absolute base pose xyzw + normalized actuator claw",
        "simulation_clock_fps": 24,
        "source_sha256": {
            name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
            for name in ("test_g1_openpi_orcalab.py", "g1_policy_client.py")
        },
        "official_success_rate": None,
        "limitations": [
            "Development joint-state proxy, not official referee.",
            "Same scene and initial pose as demonstrations; no unseen-scene generalization claim.",
            "Stripped robot and modified interactive joint limits match collection.",
            "Model actions are rate limited/normalized; all modifications are logged.",
            "Inference pauses simulation clock; measured sampling latency does not imply real-time deployment.",
            "Withdrawal/fall/causal collision/jitter are not validated by proxy evaluator.",
        ],
    }
    try:
        session.open()
        report["cameras"] = session.camera_info
        report["scene_model_sha256"] = hashlib.sha256(
            Path(session.strip.patched_path).read_bytes()
        ).hexdigest()
        report["stripped_model_applied"] = session.strip.applied
        report["disabled_collision_geometries"] = session.strip.n_col_off
        for task in tasks:
            for repeat in range(args.repeats):
                result = run_trial(
                    session,
                    client,
                    task,
                    repeat,
                    args,
                    limits,
                    args.output / f"{task}_{repeat:02d}",
                )
                records.append(result)
                if result["status"] == "failed":
                    # Preserve the failed trial and stop on infrastructure/guard failure.
                    raise RuntimeError(result["error"])
    except Exception as exc:
        report["run_error"] = f"{type(exc).__name__}: {exc}"
        print(traceback.format_exc(), flush=True)
    finally:
        try:
            session.close()
        except Exception as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        report["trials"] = records
        report["attempted_trials"] = len(records)
        report["proxy_successes"] = sum(bool(row["trial_proxy_success"]) for row in records)
        report["proxy_success_rate"] = report["proxy_successes"] / len(records) if records else None
        write_json(args.output / "report.json", report)
        print(f"Evaluation report: {args.output / 'report.json'}", flush=True)
    if "run_error" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

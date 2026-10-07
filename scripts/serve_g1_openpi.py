"""Serve the verified frozen-vision Pi05 checkpoint locally for OrcaLab tests.

Run in the independent Linux OpenPI environment. The Windows simulator sends
RGB observations over loopback; this process never connects to robot controls.
"""

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from evaluate_g1_openpi import OPENPI_COMMIT, digest, read, restore_model

MAX_REQUEST_BYTES = 16 * 1024 * 1024
CAMERAS = ("observation.images.cam_head", "observation.images.cam_wrist_r")


def decode_observation(payload):
    """Decode the small, pickle-free transport contract before GPU inference."""
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Observation exceeds request limit")
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        if set(arrays.files) != {"state", "prompt", "seed", *CAMERAS}:
            raise ValueError("Unexpected observation fields")
        state, prompt, seed = arrays["state"], arrays["prompt"], arrays["seed"]
        if state.shape != (18,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 18D state")
        if prompt.shape != () or prompt.dtype.kind != "U" or not 0 < len(str(prompt)) <= 512:
            raise ValueError("Expected a nonempty Unicode scalar prompt")
        if seed.shape != () or seed.dtype != np.uint32:
            raise ValueError("Expected a uint32 scalar seed")
        raw = {"observation.state": state.astype(np.float32), "prompt": str(prompt)}
        for name in CAMERAS:
            rgb = arrays[name]
            if (
                rgb.dtype != np.uint8
                or rgb.ndim != 3
                or rgb.shape[2] != 3
                or not 4 <= min(rgb.shape[:2]) <= max(rgb.shape[:2]) <= 2048
            ):
                raise ValueError("Expected bounded RGB uint8 HWC images")
            raw[name] = rgb.copy()
    return raw, int(seed)


class Predictor:
    def __init__(self, plan, sampling_steps):
        import jax
        from g1_openpi_data import G1Inputs, G1Outputs
        from openpi import transforms as tr
        from openpi.models import model as om
        from openpi.shared import nnx_utils, normalize
        from openpi.training.config import ModelTransformFactory

        if sys.platform != "linux" or not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("Serving requires the verified Linux JAX GPU environment")
        self.jax, self.om = jax, om
        self.model, config, _ = restore_model(plan, "finetuned")
        stats = normalize.load(Path(plan["checkpoint"]) / "assets/g1_buttons")
        self.pipeline = tr.compose(
            [
                G1Inputs(),
                tr.Normalize(stats, use_quantiles=True),
                *ModelTransformFactory()(config).inputs,
            ]
        )
        self.unnormalize = tr.Unnormalize({"actions": stats["actions"]}, use_quantiles=True)
        self.output = G1Outputs()
        self.sample = nnx_utils.module_jit(
            self.model.sample_actions, static_argnames=("num_steps",)
        )
        self.sampling_steps = sampling_steps
        self.requests = 0
        self.checkpoint = plan["checkpoint"]

    def infer(self, raw, seed):
        import jax.numpy as jnp

        tick = time.perf_counter()
        data = self.pipeline(raw)
        inputs = self.jax.tree.map(lambda value: jnp.asarray(value)[None], data)
        observation = self.om.Observation.from_dict(inputs)
        key = self.jax.random.key(seed)
        noise = self.jax.random.normal(key, (1, 24, 32))
        normalized = np.asarray(
            self.sample(key, observation, noise=noise, num_steps=self.sampling_steps)
        )[0]
        actions = self.output(self.unnormalize({"actions": normalized}))["actions"]
        elapsed = (time.perf_counter() - tick) * 1000
        self.requests += 1
        logging.info("request=%d seed=%d sampling_ms=%.1f", self.requests, seed, elapsed)
        stream = io.BytesIO()
        np.savez_compressed(
            stream, actions=actions.astype(np.float32), sampling_ms=np.float64(elapsed)
        )
        return stream.getvalue()


def make_handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, b'{"error":"unknown endpoint"}')
                return
            self.reply(
                200,
                json.dumps(
                    {
                        "ready": True,
                        "checkpoint": predictor.checkpoint,
                        "horizon": 24,
                        "action_dim": 18,
                        "fps": 24,
                        "requests": predictor.requests,
                        "sampling_steps": predictor.sampling_steps,
                        "scope": "simulation evaluation only",
                    }
                ).encode(),
            )

        def do_POST(self):
            if self.path != "/infer":
                self.reply(404, b'{"error":"unknown endpoint"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request length")
                self.connection.settimeout(30)
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("Truncated request")
                raw, seed = decode_observation(payload)
            except (ValueError, OSError, EOFError, KeyError) as error:
                self.reply(400, json.dumps({"error": str(error)}).encode())
                return
            try:
                result = predictor.infer(raw, seed)
                self.reply(200, result, "application/octet-stream")
            except Exception:
                logging.exception("Inference failed")
                self.reply(500, b'{"error":"inference failed; inspect server log"}')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, required=True, help="Verified offline evaluation plan.json"
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--sampling-steps", type=int, default=10)
    args = parser.parse_args()
    if args.sampling_steps < 1 or not 1 <= args.port <= 65535:
        parser.error("Invalid sampling steps or port")
    root = args.openpi_root.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OPENPI_COMMIT:
        raise ValueError("Expected the pinned OpenPI commit")
    plan = read(args.plan)
    checkpoint = Path(plan["checkpoint"])
    if not (checkpoint / "_CHECKPOINT_METADATA").exists():
        raise ValueError("Checkpoint is incomplete")
    if digest(checkpoint / "assets/g1_buttons/norm_stats.json") != plan["norm_stats_sha256"]:
        raise ValueError("Normalization differs from the evaluated checkpoint")
    if plan["model"]["training"]["freeze_vision"] is not True:
        raise ValueError("Expected the evaluated frozen-vision checkpoint")
    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src")]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    predictor = Predictor(plan, args.sampling_steps)
    with HTTPServer(("127.0.0.1", args.port), make_handler(predictor)) as server:
        logging.info("Ready on http://127.0.0.1:%d", args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()

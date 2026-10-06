"""Run real OpenPI Pi05 debug-network training on captured G1 data, on CPU.

Random initialization, official dummy Gemma variants, full official SigLIP.
Updates action projection/time MLP parameters only. NOT pretrained finetuning.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--openpi-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    sys.path[:0] = [
        str(args.openpi_root / "src"),
        str(args.openpi_root / "packages/openpi-client/src"),
    ]
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import sentencepiece
    from flax import nnx, serialization
    from g1_openpi_data import G1Inputs, G1Outputs, load_g1_dataset
    from openpi import transforms as tr
    from openpi.models import model as om
    from openpi.models import pi0_config, tokenizer
    from openpi.shared import normalize

    commit = subprocess.check_output(
        ["git", "-C", str(args.openpi_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if not commit.startswith("981483d"):
        raise ValueError("Expected competition-pinned OpenPI 981483d")
    print("Loading real episode and decoding both videos", flush=True)
    dataset = load_g1_dataset(args.dataset, horizon=24)
    stats = dataset.norm_stats()
    normalize.save(args.output / "assets/g1_buttons", stats)
    # Official tokenizer algorithm, with a local file instead of gs:// cache chmod on Windows.
    token_path = args.openpi_root / "g1-assets/paligemma_tokenizer.model"
    token_path.parent.mkdir(exist_ok=True)
    if not token_path.exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", token_path
        )
    tok = tokenizer.PaligemmaTokenizer.__new__(tokenizer.PaligemmaTokenizer)
    tok._max_len = 200
    tok._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(token_path))
    pipeline = tr.compose(
        [
            G1Inputs(),
            tr.Normalize(stats, use_quantiles=True),
            tr.TokenizePrompt(tok, discrete_state_input=True),
            tr.PadStatesAndActions(32),
        ]
    )
    indices = [0, min(100, len(dataset) - 1), len(dataset) - 1]
    multitask = hasattr(dataset, "sources")
    if multitask:
        indices = [
            next(
                i
                for i, (e, t) in enumerate(dataset.indices)
                if e == ep and t == (len(dataset.episodes[ep]["states"]) - 24) // 2
            )
            for ep in range(len(dataset.episodes))
        ]
    samples = [pipeline(dataset[i]) for i in indices]
    checks = {}
    checks["all_full_windows_within_episode"] = all(
        t + 24 <= len(dataset.episodes[e]["actions"]) for e, t in dataset.indices
    )
    checks["next_state_no_extra_shift"] = all(
        np.array_equal(
            dataset[i]["action"][0],
            dataset.episodes[dataset.indices[i][0]]["actions"][dataset.indices[i][1]],
        )
        for i in range(len(dataset))
    )
    checks["state_and_action_shapes"] = all(
        s["state"].shape == (32,) and s["actions"].shape == (24, 32) for s in samples
    )
    checks["finite_normalized_values"] = all(
        np.isfinite(s["state"]).all() and np.isfinite(s["actions"]).all() for s in samples
    )
    checks["padding_zero"] = all(
        np.all(s["state"][18:] == 0) and np.all(s["actions"][:, 18:] == 0) for s in samples
    )
    checks["camera_masks"] = all(
        s["image_mask"]["base_0_rgb"]
        and s["image_mask"]["right_wrist_0_rgb"]
        and not s["image_mask"]["left_wrist_0_rgb"]
        for s in samples
    )
    checks["prompt_tokenized"] = all(
        s["tokenized_prompt"].shape == (200,) and np.any(s["tokenized_prompt_mask"])
        for s in samples
    )
    # Validate runtime path accepts an observation without action labels.
    runtime = dataset[indices[1]]
    runtime.pop("action")
    checks["inference_without_labels"] = "actions" not in pipeline(runtime)
    roundtrip = tr.Unnormalize({"actions": stats["actions"]}, use_quantiles=True)(
        {"actions": samples[1]["actions"]}
    )
    checks["normalization_roundtrip"] = bool(
        np.allclose(G1Outputs()(roundtrip)["actions"], dataset[indices[1]]["action"], atol=1e-6)
    )
    from PIL import Image

    for name, rgb in samples[1]["image"].items():
        Image.fromarray(rgb).save(args.output / f"{name}.png")
    report = {
        "kind": "openpi_pi05_debug_training_chain",
        "training_chain_complete": False,
        "pretrained_weights_loaded": False,
        "production_finetuning_verified": False,
        "openpi_commit": commit,
        "dataset": str(args.dataset.resolve()),
        "fps": 24,
        "frames": dataset.info["total_frames"],
        "valid_windows": len(dataset),
        "action_horizon": 24,
        "robot_action_dim": 18,
        "model_action_dim": 32,
        "devices": [str(d) for d in jax.devices()],
        "tokenizer_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
        "checks": checks,
        "task_prompts": list(dataset.tasks.values()),
        "sources": getattr(dataset, "sources", []),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["jax", "jaxlib", "flax", "optax", "numpy", "transformers"]
        },
    }

    def save_report():
        report["elapsed_seconds"] = time.time() - started
        report["all_checks_passed"] = all(checks.values())
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save_report()
    if not all(checks.values()):
        raise AssertionError(checks)
    if args.prepare_only:
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return

    print(
        "Creating official Pi05 with dummy Gemma variants, full SigLIP; random seed 0", flush=True
    )
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=24,
        dtype="float32",
        max_token_len=200,
    )
    network = config.create(jax.random.key(0))
    # Keep full vision/language forward path, train small action-side layers on CPU.
    selected = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("PaliGemma")))
    graph, params, frozen = nnx.split(network, selected, ...)
    report["trainable_parameters"] = sum(x.size for x in jax.tree.leaves(params))
    report["frozen_parameters"] = sum(x.size for x in jax.tree.leaves(frozen) if hasattr(x, "size"))
    train_samples = samples if multitask else [samples[1]]
    batches = [jax.tree.map(lambda x: jnp.asarray(x)[None], s) for s in train_samples]
    observations = [om.Observation.from_dict(b) for b in batches]
    action_batches = [b["actions"] for b in batches]
    key = jax.random.key(42)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(train_params, frozen_params, obs, actions):
        net = nnx.merge(graph, train_params, frozen_params)
        return net.compute_loss(key, obs, actions, train=False).mean()

    value_grad = jax.jit(jax.value_and_grad(loss_fn))
    evaluate = jax.jit(loss_fn)
    before = copy.deepcopy(params.to_pure_dict())
    losses, norms = [], []
    for step in range(args.steps):
        grad = None
        task_losses = []
        for obs, actions in zip(observations, action_batches):
            value, current_grad = value_grad(params, frozen, obs, actions)
            task_losses.append(float(value))
            grad = (
                current_grad
                if grad is None
                else jax.tree.map(lambda a, b: a + b, grad, current_grad)
            )
        grad = jax.tree.map(lambda x: x / len(train_samples), grad)
        loss = float(np.mean(task_losses))
        norm = float(optax.global_norm(grad))
        if not np.isfinite(loss) or not np.isfinite(norm) or norm <= 0:
            raise AssertionError(f"Invalid loss/gradient: {loss}, {norm}")
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(loss)
        norms.append(norm)
        print(f"step={step} loss={loss:.6f} grad_norm={norm:.6f}", flush=True)
    final_loss = float(
        np.mean(
            [float(evaluate(params, frozen, o, a)) for o, a in zip(observations, action_batches)]
        )
    )
    checks["finite_nonzero_gradients"] = bool(np.all(np.isfinite(norms)) and min(norms) > 0)
    checks["parameters_updated"] = any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(params.to_pure_dict()))
    )
    checks["fixed_batch_fixed_noise_loss_decreased"] = final_loss < losses[0]
    opt_leaves, opt_tree = jax.tree.flatten(opt_state)
    checkpoint = {"params": params.to_pure_dict(), "optimizer_leaves": opt_leaves}
    encoded = serialization.to_bytes(checkpoint)
    (args.output / "debug_train_state.msgpack").write_bytes(encoded)
    restored = serialization.from_bytes(
        checkpoint, (args.output / "debug_train_state.msgpack").read_bytes()
    )
    restored_params = copy.deepcopy(params)
    restored_params.replace_by_pure_dict(jax.tree.map(jnp.asarray, restored["params"]))
    restored_loss = float(
        np.mean(
            [
                float(evaluate(restored_params, frozen, o, a))
                for o, a in zip(observations, action_batches)
            ]
        )
    )
    checks["checkpoint_loss_restored"] = bool(np.isclose(final_loss, restored_loss, atol=1e-6))
    restored_optimizer = jax.tree.unflatten(opt_tree, restored["optimizer_leaves"])
    checks["optimizer_restored"] = all(
        np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(opt_state), jax.tree.leaves(restored_optimizer))
    )
    print("Checking official action sampling and inverse normalization", flush=True)
    net = nnx.merge(graph, restored_params, frozen)
    sampler = jax.jit(net.sample_actions, static_argnames=("num_steps",))
    predicted = np.concatenate(
        [np.asarray(sampler(jax.random.key(7), o, num_steps=2)) for o in observations]
    )
    raw = G1Outputs()(
        tr.Unnormalize({"actions": stats["actions"]}, use_quantiles=True)({"actions": predicted})
    )["actions"]
    np.save(args.output / "debug_predicted_actions.npy", raw)
    checks["sampled_actions_finite_shape"] = raw.shape == (len(train_samples), 24, 18) and bool(
        np.isfinite(raw).all()
    )
    report.update(
        training_chain_complete=True,
        losses=losses,
        final_loss=final_loss,
        restored_loss=restored_loss,
        gradient_norms=norms,
        optimizer_steps=args.steps,
        microbatches_per_step=len(train_samples),
        training_scope="random-init dummy Gemma; full frozen SigLIP; action projections and time MLP only",
        checkpoint_scope="trainable parameters and optimizer; frozen random backbone recreated with seed 0; not deployable",
    )
    save_report()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise AssertionError(checks)


if __name__ == "__main__":
    main()

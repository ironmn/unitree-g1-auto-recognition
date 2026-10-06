"""Prepare or launch pretrained Pi05 LoRA using the pinned upstream train.py.

Default is configuration inspection only. --train requires the full Linux
OpenPI uv environment and enough GPU memory; not verified by the CPU smoke test.
"""

import argparse
import dataclasses
import json
import runpy
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openpi-root", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="Contains g1_buttons/norm_stats.json from smoke run",
    )
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--experiment")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error("steps and batch-size must be positive")
    commit = subprocess.check_output(
        ["git", "-C", str(args.openpi_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if not commit.startswith("981483d"):
        raise ValueError("Expected OpenPI 981483d")
    sys.path[:0] = [
        str(args.openpi_root / "src"),
        str(args.openpi_root / "packages/openpi-client/src"),
    ]
    import jax
    from g1_openpi_data import G1Inputs, G1Outputs, load_g1_dataset
    from openpi import transforms as tr
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import normalize
    from openpi.training import config as cfg
    from openpi.training import weight_loaders

    @dataclasses.dataclass(frozen=True)
    class G1DataConfig(cfg.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=tr.Group(inputs=[G1Inputs()], outputs=[G1Outputs()]),
                model_transforms=cfg.ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),
                prompt_from_task=False,
            )

    normalize.load(args.assets / "g1_buttons")
    model = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=24,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    config_name = "pi05_g1_multitask_lora" if args.dataset.is_file() else "pi05_g1_buttons_lora"
    config = cfg.TrainConfig(
        name=config_name,
        exp_name=args.experiment or config_name + "_v1",
        model=model,
        data=G1DataConfig(
            repo_id="g1_buttons_local",
            assets=cfg.AssetsConfig(assets_dir=str(args.assets.resolve()), asset_id="g1_buttons"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        batch_size=args.batch_size,
        num_workers=0,
        num_train_steps=args.steps,
        save_interval=min(100, args.steps),
        log_interval=10,
        checkpoint_base_dir=str(args.checkpoints.resolve()),
        wandb_enabled=False,
        policy_metadata={
            "fps": 24,
            "robot_action_dim": 18,
            "action_representation": "absolute_base_pose_xyzw_gripper",
        },
    )
    print(
        json.dumps(
            {
                "name": config.name,
                "checkpoint_dir": str(config.checkpoint_dir),
                "commit": commit,
                "action_dim": 32,
                "action_horizon": 24,
                "robot_dim": 18,
                "devices": [str(d) for d in jax.devices()],
                "mode": "train" if args.train else "inspect_only",
            },
            indent=2,
        )
    )
    if not args.train:
        return
    if sys.platform != "linux" or not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError(
            "Pretrained finetuning entry requires Linux + JAX GPU; use check_openpi_training.py for CPU diagnostics"
        )
    # Only replace dataset creation for this exact local repo; upstream transforms,
    # normalization, optimizer, checkpointing and train loop remain in use.
    from openpi.training import data_loader

    original = data_loader.create_torch_dataset
    dataset = load_g1_dataset(args.dataset, horizon=model.action_horizon)

    def create_dataset(data_config, action_horizon, model_config):
        if data_config.repo_id == "g1_buttons_local":
            if action_horizon != dataset.horizon:
                raise ValueError("Action horizon mismatch")
            return dataset
        return original(data_config, action_horizon, model_config)

    data_loader.create_torch_dataset = create_dataset
    try:
        entry = runpy.run_path(
            str(args.openpi_root / "scripts/train.py"), run_name="g1_openpi_train_entry"
        )
        entry["main"](config)
    finally:
        data_loader.create_torch_dataset = original


if __name__ == "__main__":
    main()

"""Restore the same pretrained values directly into final training dtypes.

Avoids retaining the entire float32 checkpoint alongside its bfloat16 copies.
The upstream shape/dtype validation, missing-LoRA initialization and train loop
are retained. Frozen/trainable filters are unchanged.
"""

import dataclasses
import logging


@dataclasses.dataclass(frozen=True)
class MemoryEfficientCheckpointWeightLoader:
    params_path: str
    restore_concurrent_gb: int = 1

    def load(self, params):
        import flax.traverse_util as traverse
        import jax
        import numpy as np
        import orbax.checkpoint as ocp
        from openpi.shared import download
        from openpi.training.weight_loaders import _merge_params

        path = download.maybe_download(self.params_path)
        reference = traverse.flatten_dict(params)

        # This pinned Orbax NumpyHandler ignores restore_concurrent_gb and
        # gathers every array read concurrently. Limit those reads explicitly.
        class SequentialNumpyHandler(ocp.type_handlers.NumpyHandler):
            async def deserialize(self, infos, args=None):
                args = args or [ocp.RestoreArgs()] * len(infos)
                result = []
                for i, (info, arg) in enumerate(zip(infos, args, strict=True)):
                    result.extend(await super().deserialize([info], [arg]))
                    if i % 10 == 0 or i == len(infos) - 1:
                        logging.info("Restored pretrained array %d/%d", i + 1, len(infos))
                return result

        registry = ocp.type_handlers.create_type_handler_registry(
            *[
                (
                    ty,
                    SequentialNumpyHandler()
                    if ty is np.ndarray
                    else ocp.type_handlers.get_type_handler(ty),
                )
                for ty in ocp.type_handlers.supported_types()
            ]
        )
        handler = ocp.PyTreeCheckpointHandler(
            restore_concurrent_gb=self.restore_concurrent_gb, type_handler_registry=registry
        )
        with ocp.Checkpointer(handler) as checkpointer:
            metadata = checkpointer.metadata(path)
            item = {"params": metadata["params"]}

            def restore_arg(key_path, leaf):
                keys = tuple(k.key for k in key_path)[1:]
                if keys[-1] == "value":
                    keys = keys[:-1]
                dtype = reference[keys].dtype if keys in reference else leaf.dtype
                return ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=dtype)

            restore_args = jax.tree_util.tree_map_with_path(restore_arg, item)
            desired_bytes = sum(
                np.prod(m.shape) * np.dtype(a.dtype).itemsize
                for m, a in zip(jax.tree.leaves(item), jax.tree.leaves(restore_args), strict=True)
            )
            logging.info("Pretrained restore output size: %.2f GiB", desired_bytes / 1024**3)
            loaded = checkpointer.restore(
                path,
                ocp.args.PyTreeRestore(item=item, restore_args=restore_args),
            )["params"]
        flat = traverse.flatten_dict(loaded)
        if all(k[-1] == "value" for k in flat):
            loaded = traverse.unflatten_dict({k[:-1]: v for k, v in flat.items()})
        logging.info("Restored pretrained weights directly into reference training dtypes")
        return _merge_params(loaded, params, missing_regex=".*lora.*")

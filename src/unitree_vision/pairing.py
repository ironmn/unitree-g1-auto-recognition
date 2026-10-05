"""Order-preserving joins of local state and camera frames."""

import math
import time
from collections import OrderedDict
from copy import deepcopy


class PairBuffer:
    """Bounded exact-index join; missing camera frames are never substituted."""

    def __init__(self, roles, timeout=3, capacity=64):
        self.roles = tuple(roles)
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("Camera roles must be nonempty and unique")
        if not math.isfinite(timeout) or timeout <= 0 or type(capacity) is not int or capacity < 1:
            raise ValueError("timeout > 0 and capacity >= 1 required")
        self.timeout, self.capacity = timeout, capacity
        self.pending = OrderedDict()
        self.dropped = 0
        self.last_index = -1

    def submit(self, state, prompt):
        index = state["simulate_index"]
        if type(index) is not int or not 0 <= index <= 2**31 - 1 or index <= self.last_index:
            raise ValueError("simulate_index must be a strictly increasing nonnegative integer")
        self.last_index = index
        self.pending[index] = {
            "state": deepcopy(state),
            "prompt": prompt,
            "images": {},
            "created": time.monotonic(),
        }
        if len(self.pending) > self.capacity:
            self.pending.popitem(last=False)
            self.dropped += 1

    def feed(self, role, frame, meta):
        if role not in self.roles:
            raise ValueError(f"Unknown camera role: {role}")
        entry = self.pending.get(meta["simulate_index"])
        if entry is not None and role not in entry["images"]:
            entry["images"][role] = (frame, dict(meta))

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        ready = []
        # Process in order: a delayed earlier camera must not reorder the dataset.
        while self.pending:
            index, entry = next(iter(self.pending.items()))
            if all(r in entry["images"] for r in self.roles):
                ready.append(entry)
                del self.pending[index]
            elif now - entry["created"] >= self.timeout:
                del self.pending[index]
                self.dropped += 1
            else:
                break
        return ready

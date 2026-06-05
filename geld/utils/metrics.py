"""Training metrics and timing utilities (adapted from LEHD)."""

import logging
import time

import numpy as np


class AverageMeter:
    """Running average of a scalar metric."""

    def __init__(self):
        """Initialize empty running average."""
        self.reset()

    def reset(self):
        """Clear accumulated sum and count."""
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        """Add a weighted observation to the running average."""
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """Return the current mean, or 0 if empty."""
        return self.sum / self.count if self.count else 0.0


class LogData:
    """Epoch-keyed metric storage for plotting."""

    def __init__(self):
        """Initialize empty epoch-keyed metric storage."""
        self.keys: set[str] = set()
        self.data: dict = {}

    def get_raw_data(self):
        """Return serializable (keys, data) for checkpointing."""
        return self.keys, self.data

    def set_raw_data(self, raw_data):
        """Restore metric storage from a checkpoint tuple."""
        self.keys, self.data = raw_data

    def append(self, key, *args):
        """Append an [epoch, value] pair to a metric series."""
        if len(args) == 1:
            args = args[0]
            if isinstance(args, (int, float)):
                value = [len(self.data[key]), args] if key in self.keys else [0, args]
            elif isinstance(args, tuple):
                value = list(args)
            elif isinstance(args, list):
                value = args
            else:
                raise ValueError("Unsupported value type")
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError("Unsupported value type")

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get(self, key):
        """Return Y values for a metric key."""
        split = np.hsplit(np.array(self.data[key]), 2)
        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        """Return X/Y lists for plotting a metric series."""
        split = np.hsplit(np.array(self.data[key]), 2)
        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()
        if not isinstance(xs, list):
            return xs, ys
        if start_idx == 0:
            return xs, ys
        if start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        raise KeyError("no start_idx value in X axis data.")

    def get_keys(self):
        """Return all recorded metric keys."""
        return self.keys

    def to_epoch_records(self) -> list[dict]:
        """Return one dict per epoch for CSV / pandas export."""
        if not self.keys:
            return []

        epochs: dict[int, dict] = {}
        for key in sorted(self.keys):
            for epoch, value in self.data[key]:
                epoch = int(epoch)
                if epoch not in epochs:
                    epochs[epoch] = {"epoch": epoch}
                epochs[epoch][key] = float(value)

        return [epochs[epoch] for epoch in sorted(epochs)]


class TimeEstimator:
    """Simple ETA estimator for long-running loops."""

    def __init__(self):
        """Initialize timer for ETA estimation."""
        self.logger = logging.getLogger("TimeEstimator")
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count: int = 1):
        """Reset timer baseline for a new loop."""
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        """Return elapsed and remaining time in hours."""
        elapsed_time = time.time() - self.start_time
        remain = total - count
        remain_time = elapsed_time * remain / max(count - self.count_zero, 1)
        return elapsed_time / 3600.0, remain_time / 3600.0

    def get_est_string(self, count, total):
        """Return formatted elapsed/remaining time strings."""
        elapsed_hours, remain_hours = self.get_est(count, total)
        elapsed_time_str = (
            f"{elapsed_hours:.2f}h" if elapsed_hours > 1.0 else f"{elapsed_hours * 60:.2f}m"
        )
        remain_time_str = (
            f"{remain_hours:.2f}h" if remain_hours > 1.0 else f"{remain_hours * 60:.2f}m"
        )
        return elapsed_time_str, remain_time_str

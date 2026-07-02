"""Device and dtype helpers."""

from collections.abc import Iterable

import torch


def setup_device(use_cuda: bool = True, cuda_device_num: int = 0) -> torch.device:
    """Select CUDA or CPU device for training and inference."""
    if use_cuda and torch.cuda.is_available():
        torch.cuda.set_device(cuda_device_num)
        return torch.device("cuda", cuda_device_num)
    return torch.device("cpu")


def move_items_to_device(items: Iterable, device: torch.device):
    """Move each tensor in an iterable to the given device."""
    return tuple(item.to(device) for item in items)


def float_dtype(device: torch.device):
    """Float tensor class for beam search on the given device."""
    return torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor


def long_dtype(device: torch.device):
    """Long tensor class for beam search indices on the given device."""
    return torch.cuda.LongTensor if device.type == "cuda" else torch.LongTensor

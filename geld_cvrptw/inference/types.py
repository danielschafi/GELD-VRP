"""Shared inference result types and decoder/post-processor base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SolveResult:
    """A complete solution. """
    tour: torch.Tensor
    length_normalized: torch.Tensor


class Decoder(ABC):
    @abstractmethod
    def decode(self, model, env) -> SolveResult:
        """Decode a tour using the given model and env. Greedy or BeamSearch here for example but could be anything"""


class PostProcessor(ABC):
    @abstractmethod
    def refine(self, model, env, result: SolveResult) -> SolveResult:
        """Method that applies some post processing to an initial solution to improve it. """

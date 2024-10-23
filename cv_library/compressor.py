from .hierarchical_compression import load_model
import torch
from typing import Literal

class Compressor():
    """Class for compressing context vectors"""
    def __init__(
        self,
        checkpoint_file: str,
        network_type: Literal['attention', 'linear'] = 'attention'
    ):
        self.network, *_ = load_model(checkpoint_file, network_type)

    def compress(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns a hierarchically-compressed version of x. Elements are
        ordered in descending order of size."""
        return self.network.forward(x)
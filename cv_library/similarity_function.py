import torch
import torch.nn as nn
import torch.nn.functional as F

def _sequence_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Normalize so that matrix multiplication becomes cos sim
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    # Find cos sim between all elements of sequence a and sequence b
    b = b.transpose(-1, -2)
    cos_sims = torch.matmul(a, b)

    # We're fitting the smaller sequence into the larger; find max along shorter dim
    if cos_sims.shape[-1] > cos_sims.shape[-2]:
        max_sims = cos_sims.max(dim=-1).values
    else:
        max_sims = cos_sims.max(dim=-2).values

    # Average the max sims and return
    return max_sims.mean(dim=-1)

class SequenceLoss(nn.Module):
    def __init__(self, y_scale: float = 1.0):
        super().__init__()
        self.y_scale = y_scale

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raw_sim_scores = _sequence_similarity(x1, x2)
        losses = (raw_sim_scores - y) ** 2

        # Scale based on the distribution of y
        scales = torch.ones_like(y)
        scales[y != 0.0] = self.y_scale
        losses = losses * scales

        return losses.sum()
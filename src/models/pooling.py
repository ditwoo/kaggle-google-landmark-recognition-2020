import torch
import torch.nn as nn
import torch.nn.functional as F


def gem(x: torch.tensor, p: int = 3, eps: float = 1e-6) -> torch.tensor:
    """Generalized Mean Pooling.

    Args:
        x (torch.tensor): input features,
            expected shapes - BxCxHxW
        p (int, optional): normalization degree.
            Defaults is `3`.
        eps (float, optional): minimum value to use in x.
            Defaults is `1e-6`.

    Returns:
        tensor with shapes - BxCx1x1
    """
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    """Generalized Mean Pooling.
    Paper: https://arxiv.org/pdf/1711.02512.
    """

    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, features):
        return gem(features, self.p, self.eps)

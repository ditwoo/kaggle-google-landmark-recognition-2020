import numpy as np
import torch


def mixup_batch(batch: torch.tensor, alphas: torch.tensor) -> torch.tensor:
    """
    Mixup batch of even indexes (0, 2, 4, ...) with batch of odd indexes (1, 3, 5, ...).

    Args:
        batch (torch.tensor): batch of data to mixup,
            tensor should have shapes - (batch_size * 2, ...)
        alphas (torch.tensor): mixup coefficient,
            tensor should have shapes - (batch_size * 2, ...)

    Returns:
        mixed torch.tensor with shape (batch_size, ...)
    """
    out = (
        batch[0::2].transpose(0, -1) * alphas[0::2]
        + batch[1::2].transpose(0, -1) * alphas[1::2]
    ).transpose(0, -1)
    return out


class Mixup:
    """Mixup coefficient generator."""

    def __init__(self, mixup_alpha: float, random_seed: int = 1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size: int) -> torch.tensor:
        """Get mixup random coefficients.

        Args:
            batch_size (int): number of elements in batch.

        Returns:
            torch.FloatTensor with mixup_lambdas with shape (batch_size,)
        """
        mixup_lambdas = []
        for _ in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1.0 - lam)
        return torch.FloatTensor(mixup_lambdas)

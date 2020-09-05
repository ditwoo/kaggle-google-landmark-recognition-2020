from torch.optim import Optimizer


def zero_grad(optimizer: Optimizer) -> None:
    """Perform an hacky way to zero gradients.
    Args:
        optimizer (Optimizer): optimizer with model parameters.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = None

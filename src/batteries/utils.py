import random
from typing import Mapping, Union, Sequence

from packaging.version import parse, Version
import numpy as np
import torch
from torch.backends import cudnn


def t2d(
    tensor: Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]],
    device: Union[str, torch.device],
) -> Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]:
    """Move tensors to a specified device.

    Args:
        tensor (Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]):
            data to move to a device.
        device (Union[str, torch.device]): device where should be moved device

    Returns:
        Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]:
            data moved to a specified device
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        # recursive move to device
        return [t2d(_tensor, device) for _tensor in tensor]
    elif isinstance(tensor, dict):
        res = {}
        for _key, _tensor in tensor.items():
            res[_key] = t2d(_tensor, device)
        return res


def seed_all(seed: int = 42) -> None:
    """Fix all seeds so results can be reproducible.

    Args:
        seed (int): random seed
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        if parse(tf.__version__) >= Version("2.0.0"):
            tf.random.set_seed(seed)
        elif parse(tf.__version__) <= Version("1.13.2"):
            tf.set_random_seed(seed)
        else:
            tf.compat.v1.set_random_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    # reproducibility
    cudnn.deterministic = True

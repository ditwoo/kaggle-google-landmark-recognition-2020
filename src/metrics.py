import torch
import numpy as np


def gap(
    predictions: torch.tensor, confidences: torch.tensor, targets: torch.tensor
) -> float:
    """Global Average Precision.

    Args:
        predictions (torch.tensor): predicted class,
            should have a size (batch,)
        confidences (torch.tensor): predicted confidence (value),
            should have a size (batch,)
        targets (torch.tensor): targets,
            should have a size (batch,)

    Returns:
        float: [description]
    """
    assert len(predictions.shape) == 1
    assert len(confidences.shape) == 1
    assert len(targets.shape) == 1
    assert predictions.shape == confidences.shape == targets.shape

    _, indices = torch.sort(confidences, descending=True)

    confidences = confidences.detach().cpu().numpy()
    predictions = predictions[indices].detach().cpu().numpy()
    targets = targets[indices].detach().cpu().numpy()

    accum, true_pos = 0.0, 0.0
    for i, (conf, pred, tgt) in enumerate(zip(confidences, predictions, targets)):
        match = int(pred == tgt)
        true_pos += match
        accum += true_pos / (i + 1) * match
    accum /= targets.shape[0]
    return accum

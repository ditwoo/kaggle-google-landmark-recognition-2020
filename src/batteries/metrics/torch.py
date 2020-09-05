import torch


def classification_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Classification accuracy metric.

    Args:
        y_true (torch.Tensor): true labels, will be flattened.
        y_pred (torch.Tensor): predicted labels, will be flattened.

    Returns:
        float: accuracy score
    """
    _true = y_true.flatten()
    _pred = y_pred.flatten()

    score = (_true == _pred).float().mean().item()

    return score


def binary_precision(
    y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5
) -> float:
    """Binary precision score.

    Args:
        y_true (torch.Tensor): true labels, will be flattened
        y_pred (torch.Tensor): logits, will be applied sigmoid, threshold and flattened.
        threshold (float, optional): threshold to apply to `y_pred`. Defaults to 0.5.

    Returns:
        float: precision score
    """
    predicted = torch.sigmoid(y_pred.flatten()) >= threshold

    if y_true.dtype in (torch.float, torch.float32, torch.float64):
        target = (y_true >= threshold).long()
    else:
        target = y_true
    target = target.flatten()

    true_positive = ((predicted == 1) & (target == 1)).float().sum().item()
    false_positive = ((predicted == 1) & (target == 0)).float().sum().item()

    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 1.0

    return precision


def binary_recall(
    y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5
) -> float:
    """Binary recall score.

    Args:
        y_true (torch.Tensor): true labels, will be flattened
        y_pred (torch.Tensor): logits, will be applied sigmoid, threshold and flattened.
        threshold (float, optional): threshold to apply to `y_pred`. Defaults to 0.5.

    Returns:
        float: precision score
    """
    predicted = torch.sigmoid(y_pred.flatten()) >= threshold

    if y_true.dtype in (torch.float, torch.float32, torch.float64):
        target = y_true >= threshold
    else:
        target = y_true
    target = target.flatten()

    true_positive = ((predicted == 1) & (target == 1)).float().sum().item()
    false_negative = ((predicted == 0) & (target == 1)).float().sum().item()

    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def binary_fbeta(
    y_true: torch.Tensor, y_pred: torch.Tensor, beta: float = 1, threshold: float = 0.5
) -> float:
    """Binary f beta score.

    Args:
        y_true (torch.Tensor): true labels, will be flattened
        y_pred (torch.Tensor): logits, will be applied sigmoid, threshold and flattened.
        beta (float, optional): [description]. Defaults to 1.
        threshold (float, optional): [description]. Defaults to 0.5.

    Returns:
        float: f beta score
    """
    predicted = torch.sigmoid(y_pred.flatten()) >= threshold

    if y_true.dtype in (torch.float, torch.float32, torch.float64):
        target = y_true >= threshold
    else:
        target = y_true
    target = target.flatten()

    true_positive = ((predicted == 1) & (target == 1)).float().sum().item()
    false_positive = ((predicted == 1) & (target == 0)).float().sum().item()
    false_negative = ((predicted == 0) & (target == 1)).float().sum().item()

    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = 0.0

    try:
        fbeta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    except ZeroDivisionError:
        fbeta = 0.0

    return fbeta

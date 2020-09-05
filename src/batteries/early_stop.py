import operator


class EarlyStopIndicator:
    def __init__(
        self,
        patience: int = 5,
        metric_minimization: bool = True,
        minimum_change: float = 0.0,
    ):
        """
        Args:
            patience (int): number of events to wait.
                Default is 5.
            metric_minimization (bool): indicator to minimize metric,
                if `True` then expected that target metric should decrease.
                Default is `True`.
            minimum_change (float): minimum change in a metric.
                Default is 0.0.
        """
        self.patience = patience
        self.minimum_change = minimum_change
        self.best_score = None
        self.counter = 0
        self.op = operator.gt if metric_minimization else operator.lt

    def __call__(self, metric: float) -> bool:
        """Check if should be executed early stopping.

        Args:
            metric (float): value of a metric at some event.

        Returns:
            `True` if should be executed early stopping,
            otherwise `False`.
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        print(
            ">>",
            metric,
            self.best_score,
            self.counter,
            self.patience,
            f"diff: {self.best_score - metric}",
            self.op(metric, self.best_score + self.minimum_change),
        )

        if self.op(metric, self.best_score + self.minimum_change):
            self.counter += 1
            return self.counter > self.patience
        else:
            self.best_score = metric
            self.counter = 0
            return False

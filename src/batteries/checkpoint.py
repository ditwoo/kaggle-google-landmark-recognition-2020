import os
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Union, Mapping, Any, Callable

import torch


def make_checkpoint(
    stage, epoch, model, optimizer=None, scheduler=None, metrics=None
) -> dict:
    """Generate checkpoint dict.

    Args:
        stage ([type]): [description]
        epoch ([type]): [description]
        model ([type]): [description]
        optimizer ([type], optional): [description]. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.
        metrics ([type], optional): [description]. Defaults to None.

    Returns:
        dict: [description]
    """
    checkpoint = {
        "stage": stage,
        "epoch": epoch,
    }
    if isinstance(model, torch.nn.DataParallel) or isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        checkpoint["model_state_dict"] = model.module.state_dict()
    else:
        checkpoint["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics
    return checkpoint


def save_checkpoint(
    checkpoint: Mapping[str, Any],
    logdir: Union[str, Path],
    name: str,
    is_best: bool = False,
    is_last: bool = False,
    verbose: bool = False,
    save_fn: Callable = torch.save,
) -> None:
    """Save checkpoint to a file.

    Args:
        checkpoint (Mapping[str, Any]): data to store in checkpoint
        logdir (Union[str, Path]): directory where should be stored checkpoint
        name (str): file name to use for storing checkpoint
        is_best (bool, optional): indicator to save checkpoint as best checkpoint.
            Defaults to False.
        is_last (bool, optional): indicator to save checkpoint as last checkpoint.
            Defaults to False.
        verbose (bool, optional): default is `False`.
        save_fn (Callable, optional): default is `torch.save`
    """
    os.makedirs(logdir, exist_ok=True)
    _name = name if name.endswith(".pth") else f"{name}.pth"
    filename = os.path.join(str(logdir), _name)
    save_fn(checkpoint, filename)
    if verbose:
        print(f"=> Saved checkpoint '{filename}'")
    if is_best:
        best_filename = os.path.join(str(logdir), "best.pth")
        shutil.copyfile(filename, best_filename)
    if is_last:
        last_filename = os.path.join(str(logdir), "last.pth")
        shutil.copyfile(filename, last_filename)


def load_checkpoint(
    checkpoint_file: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    map_location: Union[torch.device, str, Mapping[str, str], Callable] = None,
) -> None:
    """Shortcut for loading checkpoint state.

    Args:
        checkpoint_file (Union[str, Path]): path to checkpoint
        model (torch.nn.Module): model to initialize with checkpoint weights
        optimizer (torch.optim.Optimizer, optional): optimizer to initialize with checkpoint weights.
            If `None` then will be ignored.
            Default is None.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): scheduler to initialize with checkpoint weights.
            If `None` then will be ignored.
            Default is None.
        map_location (Union[torch.device, str, Mapping[int, str], Callable], optional):
            location to use for loading checkpoint content.
            More about possible locations - https://pytorch.org/docs/master/generated/torch.load.html
            Default is None.
    """
    checkpoint = torch.load(str(checkpoint_file), map_location=map_location)
    loaded_items = []

    if "model_state_dict" in checkpoint and model is not None:
        state_dict = checkpoint["model_state_dict"]
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        loaded_items.append("model")

    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loaded_items.append("optimizer")

    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        loaded_items.append("scheduler")

    if loaded_items:
        print("<= Loaded {} from '{}'".format(", ".join(loaded_items), checkpoint_file))


def checkpoints_weight_average(*files) -> OrderedDict:
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
        files: an iterable of string paths of checkpoints to load from.

    Returns:
        A dict of string keys mapping to various values. The 'model' key
        from the returned dict should correspond to an OrderedDict mapping
        string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(files)

    for f in files:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model_state_dict"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    new_state["model_state_dict"] = averaged_params
    return new_state


class CheckpointManager:
    def __init__(
        self,
        logdir: Union[str, Path],
        checkpoint_names: str = "exp",
        metric: str = "loss",
        metric_minimization: bool = True,
        save_n_best: int = 1,
        save_fn: Callable = torch.save,
    ):
        """
        Args:
            logdir (Union[str, Path]): directory where should be stored checkpoints
            checkpoint_names (str, optional): checkpoint file name.
                Default checkpoint name is "exp".
            metric (str, optional): metric name.
                Default is "loss".
            metric_minimization (bool, optional): indicator to minimize metric,
                if `True` then expected that target metric should decrease.
                Default is True.
            save_n_best (int, optional): number of best checkpoints to keep.
                Default is 1.
            save_fn (Callable, optional): default is `torch.save`
        """
        self.logdir = logdir
        self.checkpoint_filename = checkpoint_names
        self.metric_name = metric
        self.metric_minimization = metric_minimization
        self.save_n_best = save_n_best
        self.metrics = []
        self.best_metrics = []
        self.save_fn = save_fn

    def _save_metrics(self) -> None:
        with open(os.path.join(self.logdir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def checkpoint_name(self, epoch: int) -> str:
        return f"{self.checkpoint_filename}_{epoch}.pth"

    def process(
        self, metric_value: float, epoch: int, checkpoint: Mapping[str, Any]
    ) -> None:
        """Generate checkpoint file and store only required checkpoints.

        Args:
            metric_value (float): value of a target metric
            epoch (int): epoch index
            checkpoint (Mapping[str, Any]): data to store in a checkpoint file
        """
        # determine arguments for save method
        if len(self.metrics):
            last_best_score = sorted(
                self.metrics,
                key=lambda record: record[self.metric_name],
                reverse=not self.metric_minimization,
            )[0][self.metric_name]
            if self.metric_minimization:
                is_best = metric_value <= last_best_score
            else:
                is_best = metric_value >= last_best_score
        else:
            is_best = True
        # store checkpoint
        checkpoint_name = self.checkpoint_name(epoch)
        save_checkpoint(
            checkpoint=checkpoint,
            logdir=self.logdir,
            name=checkpoint_name,
            is_best=is_best,
            is_last=True,
            save_fn=self.save_fn,
        )
        # update metrics
        metric_record = {"epoch": epoch, self.metric_name: metric_value}
        self.metrics.append(metric_record)
        self.best_metrics.append(metric_record)
        # remove old not required checkpoint
        if len(self.best_metrics) > self.save_n_best:
            self.best_metrics = sorted(
                self.best_metrics,
                key=lambda record: record[self.metric_name],
                reverse=not self.metric_minimization,
            )
            to_remove = os.path.join(
                self.logdir, self.checkpoint_name(self.best_metrics.pop(-1)["epoch"])
            )
            os.remove(to_remove)
        # overwrite existing metrics
        self._save_metrics()

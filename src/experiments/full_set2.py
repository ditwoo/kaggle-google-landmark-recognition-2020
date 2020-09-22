import os
import shutil
import time
from pathlib import Path

# installed
import numpy as np
import pandas as ps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import albumentations as albu
from albumentations.pytorch import ToTensorV2

# local
from src.metrics import gap
from src.models import EncoderWithHead
from src.models.efficientnets import EfficientNetEncoder
from src.models.heads import CosFace
from src.datasets import FolderDataset, LimitedClassSampler
from src.batteries import (
    seed_all,
    zero_grad,
    CheckpointManager,
    TensorboardLogger,
    t2d,
    make_checkpoint,
    load_checkpoint,
)
from src.batteries.progress import tqdm


torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEBUG = int(os.environ.get("DEBUG", -1))
EMBEDDING_SIZE = 512
NUM_CLASSESS = 81313
DATA_DIR = Path(".") / "input"
TRAIN_VALID_FILE = DATA_DIR / "train_valid.pkl"
IMAGES_DIR = DATA_DIR / "train"
NUM_WORKERS = 26


def get_loaders(stage: str, train_bs: int = 32, valid_bs: int = 64) -> tuple:
    """Prepare loaders for a stage.

    Args:
        stage (str): stage name
        train_bs (int, optional): batch size for training dataset.
            Default is `32`.
        valid_bs (int, optional): batch size for validation dataset.
            Default is `64`.

    Returns:
        train and validation data loaders
    """

    train_valid = ps.read_pickle(TRAIN_VALID_FILE)
    train = train_valid[train_valid["is_valid"] == False]
    valid = train_valid[train_valid["is_valid"] == True]

    landmark_map = {
        landmark: idx
        for idx, landmark in enumerate(sorted(set(train_valid["landmark_id"].values)))
    }

    train_augs = albu.Compose(
        [
            albu.RandomResizedCrop(224, 224, scale=(0.6, 1.0)),
            albu.HorizontalFlip(p=0.5),
            # albu.JpegCompression(p=0.5),
            albu.Normalize(),
            ToTensorV2(),
        ]
    )
    valid_augs = albu.Compose([albu.Resize(224, 224), albu.Normalize(), ToTensorV2(),])

    train_set = FolderDataset(
        train["id"].values,
        train["landmark_id"].values,
        landmark_map,
        transforms=train_augs,
        data_dir=IMAGES_DIR,
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_bs,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )
    print(
        f" * Num records in train dataset - {len(train_set)}, batches - {len(train_loader)}"
    )

    valid_set = FolderDataset(
        valid["id"].values,
        valid["landmark_id"].values,
        landmark_map,
        transforms=valid_augs,
        data_dir=IMAGES_DIR,
    )
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=valid_bs, num_workers=NUM_WORKERS
    )
    print(
        f" * Num records in valid dataset - {len(valid_set)}, batches - {len(valid_loader)}"
    )

    return train_loader, valid_loader


def train_fn(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler=None,
    accumulation_steps: int = 1,
    verbose: bool = True,
) -> dict:
    """Train step.

    Args:
        model (nn.Module): model to train
        loader (DataLoader): loader with data
        device (str): device to use for placing batches
        loss_fn (nn.Module): loss function, should be callable
        optimizer (optim.Optimizer): model parameters optimizer
        scheduler ([type], optional): batch scheduler to use.
            Default is `None`.
        accumulation_steps (int, optional): number of steps to accumulate gradients.
            Default is `1`.
        verbose (bool, optional): verbosity mode.
            Default is True.

    Returns:
        dict with metics computed during the training on loader
    """
    model.train()

    metrics = {
        "loss": [],
        "gap": [],
        "accuracy": [],
    }

    with tqdm(total=len(loader), desc="train", disable=not verbose) as progress:
        for _idx, batch in enumerate(loader):
            inputs, targets = t2d(batch, device)

            zero_grad(optimizer)

            outputs = model(inputs, targets)
            loss = loss_fn(outputs, targets)

            _loss = loss.detach().item()
            metrics["loss"].append(_loss)

            classes = torch.argmax(outputs, 1)
            _acc = (classes == targets).float().mean().detach().item()
            metrics["accuracy"].append(_acc)

            confidences, predictions = torch.max(outputs, dim=1)
            _gap = gap(predictions, confidences, targets)
            metrics["gap"].append(_gap)

            loss.backward()

            progress.set_postfix_str(
                f"loss {_loss:.4f}, gap {_gap:.4f}, accuracy {_acc:.4f}"
            )

            if (_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            progress.update(1)

            if _idx == DEBUG:
                break

    metrics["loss"] = np.mean(metrics["loss"])
    metrics["gap"] = np.mean(metrics["gap"])
    metrics["accuracy"] = np.mean(metrics["accuracy"])
    return metrics


def valid_fn(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    loss_fn: nn.Module,
    verbose: bool = True,
) -> dict:
    """Validation step.

    Args:
        model (nn.Module): model to train
        loader (DataLoader): loader with data
        device (str): device to use for placing batches
        loss_fn (nn.Module): loss function, should be callable
        verbose (bool, optional): verbosity mode.
            Default is True.

    Returns:
        dict with metics computed during the validation on loader
    """
    model.eval()

    metrics = {
        "loss": [],
        "gap": [],
        "accuracy": [],
    }

    with torch.no_grad(), tqdm(
        total=len(loader), desc="valid", disable=not verbose
    ) as progress:
        for _idx, batch in enumerate(loader):
            inputs, targets = t2d(batch, device)

            outputs = model(inputs, targets)
            loss = loss_fn(outputs, targets)

            _loss = loss.detach().item()
            metrics["loss"].append(_loss)

            classes = torch.argmax(outputs, 1)
            _acc = (classes == targets).float().mean().detach().item()
            metrics["accuracy"].append(_acc)

            confidences, predictions = torch.max(outputs, dim=1)
            _gap = gap(predictions, confidences, targets)
            metrics["gap"].append(_gap)

            progress.set_postfix_str(
                f"loss {_loss:.4f}, gap {_gap:.4f}, accuracy - {_acc}"
            )
            progress.update(1)

            if _idx == DEBUG:
                break

    metrics["loss"] = np.mean(metrics["loss"])
    metrics["gap"] = np.mean(metrics["gap"])
    metrics["accuracy"] = np.mean(metrics["accuracy"])
    return metrics


def log_metrics(
    stage: str, metrics: dict, logger: TensorboardLogger, loader: str, epoch: int
) -> None:
    """Write metrics to tensorboard and stdout.

    Args:
        stage (str): stage name
        metrics (dict): metrics computed during training/validation steps
        logger (TensorboardLogger): logger to use for storing metrics
        loader (str): loader name
        epoch (int): epoch number
    """
    order = ("loss", "gap", "accuracy")
    for metric_name in order:
        if metric_name in metrics:
            value = metrics[metric_name]
            logger.metric(f"{stage}/{metric_name}", {loader: value}, epoch)
            print(f"{metric_name:>10}: {value:.4f}")
    print()


def experiment(logdir: Path, device: torch.device) -> None:
    """Experiment function

    Args:
        logdir (Path): directory where should be placed logs
        device (str): device name to use
    """
    tb_dir = logdir / "tensorboard"
    main_metric = "gap"
    minimize_metric = False

    seed_all()

    model = EncoderWithHead(
        EfficientNetEncoder("efficientnet-b0", EMBEDDING_SIZE, bias=False),
        CosFace(EMBEDDING_SIZE, NUM_CLASSESS, None),
    )
    load_checkpoint("./logs/full_set/stage_0/last.pth", model)

    model.head.s = np.sqrt(2) * np.log(NUM_CLASSESS - 1)
    model.head.m = 0

    # # freeze backbone of encoder
    # for parameter in model.encoder.base._blocks.parameters():
    #     parameter.requires_grad = False

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5,)
    criterion = nn.CrossEntropyLoss()
    scheduler = None

    with TensorboardLogger(tb_dir) as tb:

        stage = "stage_0"
        n_epochs = 10
        print(f"Stage - '{stage}'")

        checkpointer = CheckpointManager(
            logdir=logdir / stage,
            metric=main_metric,
            metric_minimization=minimize_metric,
            save_n_best=5,
        )

        train_loader, valid_loader = get_loaders(stage, train_bs=128, valid_bs=128)

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{epoch_start_time}]\n[Epoch {epoch}/{n_epochs}]")

            train_metrics = train_fn(model, train_loader, device, criterion, optimizer)
            log_metrics(stage, train_metrics, tb, "train", epoch)

            valid_metrics = valid_fn(model, valid_loader, device, criterion)
            log_metrics(stage, valid_metrics, tb, "valid", epoch)

            checkpointer.process(
                metric_value=valid_metrics[main_metric],
                epoch=epoch,
                checkpoint=make_checkpoint(
                    stage,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    metrics={"train": train_metrics, "valid": valid_metrics},
                ),
            )


def main() -> None:
    experiment_name = "full_set2"
    logdir = Path(".") / "logs" / experiment_name

    if not torch.cuda.is_available():
        raise ValueError("Something went wrong - CUDA devices is not available!")

    device = torch.device("cuda:0")

    if logdir.is_dir():
        shutil.rmtree(logdir, ignore_errors=True)
        print(f" * Removed existing directory with logs - '{logdir}'")

    experiment(logdir, device)


if __name__ == "__main__":
    main()

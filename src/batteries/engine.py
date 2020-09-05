import numpy as np
import sys

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import t2d


def train_fn(
    model: nn.Module,
    loader,
    device,
    loss_fn,
    optimizer,
    scheduler=None,
    accum_steps: int = 1,
    verbose=True,
):
    model.train()

    losses = []
    prbar = tqdm(enumerate(loader), file=sys.stdout, desc="train")
    for _idx, (bx, by) in prbar:
        bx = t2d(bx, device)
        by = t2d(by, device)

        optimizer.zero_grad()

        if isinstance(bx, (tuple, list)):
            outputs = model(*bx)
        elif isinstance(bx, dict):
            outputs = model(**bx)
        else:
            outputs = model(bx)

        loss = loss_fn(outputs, by)
        losses.append(loss.item())
        loss.backward()

        prbar.set_postfix_str(f"loss {loss:.4f}")

        if (_idx + 1) % accum_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    return np.mean(losses)


def valid_fn(model: nn.Module, loader, device, loss_fn):
    model.eval()

    losses = []
    with torch.no_grad():
        for bx, by in tqdm(loader, file=sys.stdout, desc="valid"):
            bx = t2d(bx, device)
            by = t2d(by, device)

            if isinstance(bx, (tuple, list)):
                outputs = model(*bx)
            elif isinstance(bx, dict):
                outputs = model(**bx)
            else:
                outputs = model(bx)

            loss = loss_fn(outputs, by)
            losses.append(loss.item())

    return np.mean(losses)

import pandas as ps
import random
from torch.utils.data import Dataset, Sampler
from typing import Sequence


class LimitedClassSampler(Sampler):
    def __init__(self, targets: Sequence[int], max_samples: int = 100):
        self.max_samples = max_samples
        self.groups = (
            ps.DataFrame(targets, columns=["targets"])
            .reset_index()
            .groupby("targets")
            .agg({"index": list})
            .reset_index()
        )
        num_samples = self.groups["index"].apply(len)
        self.size = (
            num_samples[num_samples < max_samples].values.sum()  # < max_samples
            + (num_samples >= max_samples).values.sum() * max_samples  # >= max samples
        )

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        indices = []
        for idx in self.groups.index:
            class_samples = self.groups.at[idx, "index"]
            if len(class_samples) < self.max_samples:
                indices.extend(class_samples)
            else:
                indices.extend(random.sample(class_samples, self.max_samples))
        return iter(indices)


class SampledDataset(Dataset):
    """Dataset and LimitedClassSampler as one object."""

    def __init__(self, dataset: Dataset, max_samples: int = 100):
        """
        Args:
            dataset (Dataset): dataset to wrap
            max_samples (int, optional): number of samples per class.
                Default is 100.
        """
        self.dataset = dataset
        self.max_samples = max_samples
        if not hasattr(dataset, "targets"):
            raise ValueError(
                "dataset does not have a method 'targets' (get list of lables)!"
            )
        self.groups = (
            ps.DataFrame(dataset.targets(), columns=["targets"])
            .reset_index()
            .groupby("targets")
            .agg({"index": list})
            .reset_index()
        )
        self.groups["samples"] = self.groups["index"].apply(len)
        self.groups.at[self.groups["samples"] >= 100, "samples"] = 100
        self.size = self.groups["samples"].values.sum()
        self.indices = None
        self._iter_indices()

    def _iter_indices(self) -> None:
        self.indices = []
        for idx in self.groups.index:
            class_samples = self.groups.at[idx, "index"]
            if len(class_samples) < self.max_samples:
                self.indices.extend(class_samples)
            else:
                self.indices.extend(random.sample(class_samples, self.max_samples))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

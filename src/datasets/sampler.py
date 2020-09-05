import pandas as ps
import random
from torch.utils.data import Sampler
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


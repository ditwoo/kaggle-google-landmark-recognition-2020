from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Sequence, Mapping
from albumentations import BasicTransform, Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


DEFAULT_TRANSFORM = Compose([Resize(224, 224), Normalize(), ToTensorV2()])


class FolderDataset(Dataset):
    def __init__(
        self,
        file_ids: Sequence[str],
        landmark_ids: Sequence[str],
        landmark_map: Mapping[int, int],
        transforms: BasicTransform = DEFAULT_TRANSFORM,
        data_dir: Path = Path("."),
    ):
        self.data_dir = data_dir
        self.file_ids = file_ids
        self.landmark_ids = landmark_ids
        self.landmark_map = landmark_map
        self.transforms = transforms

        if len(landmark_ids) != len(file_ids):
            raise ValueError(
                "Different number of files ({}) and lanmarks ({})!".format(
                    len(file_ids), len(landmark_ids)
                )
            )

    def __len__(self) -> int:
        return len(self.file_ids)

    def landmark_to_vec(self, landmark_id: int) -> np.ndarray:
        vec = np.zeros(len(self.landmark_map), dtype="f")
        vec[self.landmark_map[landmark_id]] = 1
        return vec

    def landmark_to_class(self, landmark_id: int) -> torch.tensor:
        vec = torch.tensor(self.landmark_map[landmark_id], dtype=torch.long)
        return vec

    def __getitem__(self, idx: int) -> int:
        file = self.file_ids[idx]
        file = Path(file[0]) / file[1] / file[2] / f"{file}.jpg"
        if self.data_dir is not None:
            file = self.data_dir / file
        file = str(file)

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        landmark = self.landmark_ids[idx]
        landmark_vec = self.landmark_to_class(landmark)

        return img, landmark_vec


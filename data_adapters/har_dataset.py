import os

import numpy as np
import torch
from torch.utils import data


HAR_CLASSES = [
    "walk",
    "upstairs",
    "downstairs",
    "sit",
    "stand",
    "lie",
]


class HARTFDAData(data.Dataset):
    """
    Adapter for TFDA-style HAR .pt files.

    Expected files:
      - train_<domain>.pt
      - test_<domain>.pt

    Expected fields:
      - samples: (N, C, L), (N, L, C), or (N, L)
      - labels:  (N,)

    Returned samples follow the TimeMatch protocol:
      pixels:       (T, C, 1)
      valid_pixels: (T, 1)
      positions:    0..T-1
      extra:        empty vector
      label:        int in [0, 5]
    """

    classes = HAR_CLASSES

    def __init__(
        self,
        data_root,
        domain,
        split="train",
        transform=None,
        indices=None,
        input_dim=9,
        label_offset="auto",
    ):
        super().__init__()
        self.data_root = data_root
        self.domain = str(domain)
        self.dataset_name = self.domain
        self.country = "har"
        self.split = split
        self.transform = transform
        self.input_dim = int(input_dim)
        self.with_extra = False
        self.closed_set = True

        path = os.path.join(data_root, f"{split}_{self.domain}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"HAR TFDA file not found: {path}")

        dataset = torch.load(path, map_location="cpu", weights_only=False)
        if "samples" not in dataset or "labels" not in dataset:
            raise KeyError(f"{path} must contain 'samples' and 'labels' fields")

        samples = dataset["samples"]
        labels = dataset["labels"]
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        samples = samples.float()
        if samples.dim() == 2:
            samples = samples.unsqueeze(1)
        elif samples.dim() == 3 and samples.shape[1] != self.input_dim:
            samples = samples.transpose(1, 2)
        if samples.dim() != 3:
            raise ValueError(f"Expected HAR samples with 2 or 3 dims, got {tuple(samples.shape)}")
        if samples.shape[1] != self.input_dim:
            raise ValueError(
                f"HAR input_dim mismatch: samples have {samples.shape[1]} channels, "
                f"but input_dim={self.input_dim}"
            )

        labels = labels.long().view(-1)
        if label_offset == "auto":
            if labels.numel() > 0 and int(labels.min()) == 1 and int(labels.max()) <= len(HAR_CLASSES):
                labels = labels - 1
        elif int(label_offset) != 0:
            labels = labels - int(label_offset)
        if labels.numel() > 0 and (int(labels.min()) < 0 or int(labels.max()) >= len(HAR_CLASSES)):
            raise ValueError(
                f"HAR labels must map to [0, {len(HAR_CLASSES) - 1}], "
                f"got range [{int(labels.min())}, {int(labels.max())}]"
            )

        if indices is not None:
            indices = sorted(int(idx) for idx in indices)
            samples = samples[indices]
            labels = labels[indices]
            self.indices = indices
        else:
            self.indices = list(range(samples.shape[0]))

        self.samples_tensor = samples.contiguous()
        self.labels = labels.contiguous()
        self.date_positions = np.arange(samples.shape[-1], dtype=np.int64)
        self.date_indices = np.arange(samples.shape[-1], dtype=np.int64)
        self.dates = [str(int(pos)) for pos in self.date_positions]

    def __len__(self):
        return int(self.samples_tensor.shape[0])

    def get_labels(self):
        return self.labels.cpu().numpy()

    def get_shapes(self):
        time_steps = int(self.samples_tensor.shape[-1])
        channels = int(self.samples_tensor.shape[1])
        return [(time_steps, channels, 1) for _ in range(len(self))]

    def __getitem__(self, index):
        x = self.samples_tensor[index].cpu().numpy()  # (C, L)
        pixels = np.transpose(x, (1, 0))[:, :, np.newaxis]  # (T, C, 1)
        sample = {
            "index": int(index),
            "parcel_index": int(self.indices[index]),
            "pixels": pixels.astype(np.float32),
            "valid_pixels": np.ones((pixels.shape[0], 1), dtype=np.float32),
            "positions": self.date_positions.copy(),
            "extra": np.zeros((0,), dtype=np.float32),
            "label": int(self.labels[index].item()),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


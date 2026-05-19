from collections import Counter
from copy import deepcopy

import torch
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import transforms

from dataset import (
    GroupByShapesBatchSampler,
    PixelSetData,
    create_train_loader,
)
from data_adapters.har_dataset import HARTFDAData, get_adatime_classes, get_adatime_input_dim
from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    RandomTemporalShift,
    ToTensor,
)
from utils import label_utils


def get_dataset_type(config):
    return str(getattr(config, "dataset_type", "remote_sensing")).lower()


def is_har(config):
    return get_dataset_type(config) == "har"


class RandomSampleTimeStepsIfLonger:
    """Sample time steps only when the sequence is longer than the requested length."""

    def __init__(self, seq_length):
        self.seq_length = int(seq_length)

    def __call__(self, sample):
        if self.seq_length <= 0:
            return sample
        if sample["pixels"].shape[0] <= self.seq_length:
            return sample
        return RandomSampleTimeSteps(self.seq_length)(sample)


def get_classes_for_config(config):
    if is_har(config):
        return get_adatime_classes(getattr(config, "har_dataset_name", "HAR"))
    source_classes = label_utils.get_classes(
        str(config.source).split("/")[0],
        combine_spring_and_winter=getattr(config, "combine_spring_and_winter", False),
    )
    if getattr(config, "closed_set", False):
        source_classes = [cls for cls in source_classes if cls != "unknown"]
    source_data = build_dataset(config, config.source, source_classes, split="train")
    labels, counts = torch.unique(torch.as_tensor(source_data.get_labels()), return_counts=True)
    keep = {int(label.item()) for label, count in zip(labels, counts) if int(count.item()) >= 200}
    return [cls for idx, cls in enumerate(source_classes) if idx in keep]


def build_dataset(config, dataset_name, classes=None, transform=None, indices=None, split="train"):
    if is_har(config):
        adatime_dataset = getattr(config, "har_dataset_name", "HAR")
        input_dim = getattr(config, "input_dim", None)
        if input_dim is None or int(input_dim) == 10:
            input_dim = get_adatime_input_dim(adatime_dataset)
        return HARTFDAData(
            config.data_root,
            dataset_name,
            split=split,
            transform=transform,
            indices=indices,
            input_dim=input_dim,
            label_offset=getattr(config, "har_label_offset", "auto"),
            adatime_dataset=adatime_dataset,
        )
    return PixelSetData(
        config.data_root,
        dataset_name,
        classes if classes is not None else config.classes,
        transform,
        indices=indices,
        closed_set=getattr(config, "closed_set", False),
    )


def get_dataset_length(config, dataset_name, split="train"):
    return len(build_dataset(config, dataset_name, getattr(config, "classes", None), split=split))


def make_train_transform(config):
    if is_har(config):
        return transforms.Compose(
            [
                RandomSamplePixels(config.num_pixels),
                RandomSampleTimeStepsIfLonger(config.seq_length),
                ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p)
            if config.with_shift_aug
            else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )


def make_weak_transform(config):
    if is_har(config):
        return transforms.Compose([RandomSamplePixels(config.num_pixels), ToTensor()])
    return transforms.Compose([RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()])


def make_strong_transform(config):
    if is_har(config):
        return transforms.Compose(
            [
                RandomSamplePixels(config.num_pixels),
                RandomSampleTimeStepsIfLonger(config.seq_length),
                ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            Normalize(),
            ToTensor(),
        ]
    )


def make_eval_transform(config, sample_pixels=False, sample_time=False):
    if is_har(config):
        steps = [RandomSamplePixels(config.num_pixels) if sample_pixels else Identity()]
        steps.append(RandomSampleTimeStepsIfLonger(config.seq_length) if sample_time else Identity())
        steps.append(ToTensor())
        return transforms.Compose(steps)
    return transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels) if sample_pixels else Identity(),
            RandomSampleTimeSteps(config.seq_length) if sample_time else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )


def create_train_dataset(config, dataset_name, splits, transform=None):
    return build_dataset(
        config,
        dataset_name,
        config.classes,
        transform if transform is not None else make_train_transform(config),
        indices=splits[dataset_name]["train"],
        split="train",
    )


def create_evaluation_loaders_for_config(dataset_name, splits, config, sample_pixels_val=False):
    is_tsnet = config.model == "tsnet"
    val_transform = make_eval_transform(
        config,
        sample_pixels=sample_pixels_val,
        sample_time=is_tsnet,
    )
    test_transform = make_eval_transform(config, sample_pixels=False, sample_time=is_tsnet)

    val_dataset = build_dataset(
        config,
        dataset_name,
        config.classes,
        val_transform,
        indices=splits[dataset_name]["val"],
        split="train",
    )
    if is_har(config):
        test_dataset = build_dataset(
            config,
            dataset_name,
            config.classes,
            test_transform,
            indices=None,
            split="test",
        )
        val_loader = data.DataLoader(
            val_dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = data.DataLoader(
            test_dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        val_loader = data.DataLoader(
            val_dataset,
            num_workers=config.num_workers,
            batch_sampler=GroupByShapesBatchSampler(
                val_dataset,
                config.batch_size,
                by_pixel_dim=not sample_pixels_val,
            ),
        )
        test_dataset = build_dataset(
            config,
            dataset_name,
            config.classes,
            test_transform,
            indices=splits[dataset_name]["test"],
            split="train",
        )
        test_loader = data.DataLoader(
            test_dataset,
            num_workers=config.num_workers,
            batch_sampler=GroupByShapesBatchSampler(test_dataset, config.batch_size),
        )

    print("evaluation dataset:", dataset_name)
    print(f"val target data: {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"test taget data: {len(test_dataset)} ({len(test_loader)} batches)")
    return val_loader, test_loader


def create_training_loader(dataset, config):
    return create_train_loader(dataset, config.batch_size, config.num_workers)


def create_timematch_data_loaders(splits, config, tuple_dataset_cls, balance_source=True):
    weak_aug = make_weak_transform(config)
    strong_aug = make_strong_transform(config)

    source_dataset = build_dataset(
        config,
        config.source,
        config.classes,
        strong_aug,
        indices=splits[config.source]["train"],
        split="train",
    )

    if balance_source:
        source_labels = source_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

    target_dataset = build_dataset(
        config,
        config.target,
        config.classes,
        None,
        indices=splits[config.target]["train"],
        split="train",
    )
    strong_dataset = deepcopy(target_dataset)
    strong_dataset.transform = strong_aug
    weak_dataset = deepcopy(target_dataset)
    weak_dataset.transform = weak_aug
    target_dataset_weak_strong = tuple_dataset_cls(weak_dataset, strong_dataset)

    no_aug_dataset = deepcopy(target_dataset)
    no_aug_dataset.transform = weak_aug
    target_loader_no_aug = data.DataLoader(
        no_aug_dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    target_loader_weak_strong = data.DataLoader(
        target_dataset_weak_strong,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    print(f"size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)")
    print(f"size of target dataset: {len(target_dataset)} ({len(target_loader_weak_strong)} batches)")
    return source_loader, target_loader_no_aug, target_loader_weak_strong

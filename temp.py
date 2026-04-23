from collections import defaultdict

import numpy as np


def cal(dataloader):
    """
    Compute source-domain class separability.

    Larger values indicate that class prototypes are more distinct within
    the source domain, which usually makes the source domain easier to learn.
    """
    class_doy_means = compute_class_doy_means(
        collect_class_doy_parcel_means(dataloader)
    )
    return compute_separability(class_doy_means)


def collect_class_doy_parcel_means(dataloader):
    """
    Collect parcel-level observations grouped by class and day-of-year.

    For each sample and timestamp, only valid pixels are used. Pixel values
    are averaged within the parcel first so that parcels with more valid
    pixels do not dominate the class prototype.
    """
    data_dict = defaultdict(lambda: defaultdict(list))

    for batch in dataloader:
        pixels = batch["pixels"]  # [B, T, C, S]
        valid_pixels = batch["valid_pixels"]  # [B, T, S]
        positions = batch["positions"]  # [B, T]
        labels = batch["label"]  # [B]

        batch_size, num_steps = pixels.shape[:2]

        for i in range(batch_size):
            class_id = labels[i].item()

            for t in range(num_steps):
                doy = positions[i, t].item()
                if doy <= 0:
                    continue

                pixel_values_at_t = pixels[i, t]  # [C, S]
                valid_mask_at_t = valid_pixels[i, t] > 0.5  # [S]

                if valid_mask_at_t.sum().item() == 0:
                    continue

                parcel_mean = (
                    pixel_values_at_t[:, valid_mask_at_t].mean(dim=1).cpu().numpy()
                )
                data_dict[class_id][doy].append(parcel_mean)

    return data_dict


def compute_class_doy_means(data_dict):
    """
    Compute one class prototype vector for each (class_id, doy).
    """
    class_doy_means = {}

    for class_id, doy_dict in data_dict.items():
        class_doy_means[class_id] = {}
        for doy, vectors in doy_dict.items():
            if vectors:
                class_doy_means[class_id][doy] = np.mean(np.stack(vectors), axis=0)

    return class_doy_means


def compute_pairwise_structure(class_doy_means):
    """
    Build the pairwise class distance matrix based on average MSE over common doy.

    Returns:
        classes: sorted class ids
        dist_mat: [C, C] symmetric matrix
        pair_scores: dict[(class_a, class_b)] -> float
    """
    classes = sorted(class_doy_means.keys())
    num_classes = len(classes)
    dist_mat = np.zeros((num_classes, num_classes), dtype=np.float32)
    pair_scores = {}

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            class_a = classes[i]
            class_b = classes[j]

            common_doy = sorted(
                set(class_doy_means[class_a].keys())
                & set(class_doy_means[class_b].keys())
            )

            if not common_doy:
                continue

            doy_scores = []
            for doy in common_doy:
                mean_a = class_doy_means[class_a][doy]
                mean_b = class_doy_means[class_b][doy]
                doy_scores.append(float(np.mean((mean_a - mean_b) ** 2)))

            pair_score = float(np.mean(doy_scores))
            dist_mat[i, j] = pair_score
            dist_mat[j, i] = pair_score
            pair_scores[(class_a, class_b)] = pair_score

    return classes, dist_mat, pair_scores


def compute_separability(class_doy_means):
    """
    Compute source-domain class separability from pairwise class distances.

    Separability = mean pairwise MSE between class prototypes, averaged over
    the day-of-year values shared by each class pair.

    Larger values mean higher class distinctiveness and lower ambiguity
    inside the source domain.
    """
    _, _, pair_scores = compute_pairwise_structure(class_doy_means)

    if not pair_scores:
        return 0.0

    return float(np.mean(list(pair_scores.values())))

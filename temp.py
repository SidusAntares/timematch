from collections import defaultdict
import numpy as np
import torch
import math

def cal(dataloader):
    return compute_separability(compute_class_doy_means(collect_class_doy_data(dataloader)))

def collect_class_doy_data(dataloader):
    """
    收集所有有效像素的观测，按类别和doy分组
    """
    data_dict = defaultdict(lambda: defaultdict(lambda: {'values': [], 'count': 0}))

    for batch in dataloader:
        pixels = batch['pixels']  # [B, T, C, S]
        valid_pixels = batch['valid_pixels']  # [B, T, S]
        positions = batch['positions']  # [B, T]
        labels = batch['label']  # [B]

        for i in range(pixels.shape[0]):
            class_id = labels[i].item()

            for t in range(pixels.shape[1]):
                current_doy = positions[i, t].item()

                if current_doy <= 0:
                    continue

                # 收集所有有效像素
                if pixels.dim() == 4:
                    pixel_values_at_t = pixels[i, t]  # [C, S]
                    valid_mask_at_t = valid_pixels[i, t]  # [S]

                    for p in range(pixel_values_at_t.shape[1]):
                        if valid_mask_at_t[p] > 0.5:
                            spectral_values = pixel_values_at_t[:, p].cpu().numpy()
                            data_dict[class_id][current_doy]['values'].append(spectral_values)
                            data_dict[class_id][current_doy]['count'] += 1

    return data_dict


def compute_class_doy_means(data_dict):
    """
    计算类中心：对每个 (class_id, doy) 组合的所有像素取平均
    """
    class_doy_means = {}

    for class_id, doy_dict in data_dict.items():
        class_doy_means[class_id] = {}

        for doy, data in doy_dict.items():
            if data['count'] > 0:
                # 对所有像素取平均（包括不同parcel的像素）
                all_pixels = np.stack(data['values'])  # [N_pixels, D]
                mean_vector = np.mean(all_pixels, axis=0)  # [D]
                class_doy_means[class_id][doy] = mean_vector

    return class_doy_means


def compute_separability(class_doy_means):
    """
    计算 Separability 指标

    正确公式:
    Separability = 2/(C(C-1)) * Σ_{A=1}^C Σ_{B=A+1}^C [1/(T×D) * Σ_{t=1}^T Σ_{d=1}^D |μ_A(t,d) - μ_B(t,d)|]

    关键点:
    1. 使用绝对值 |μ_A - μ_B|，避免正负抵消
    2. 只对两个类别都有观测的doy进行计算
    """
    classes = sorted(class_doy_means.keys())
    C = len(classes)

    if C < 2:
        return 0.0

    total_pairwise_diff = 0.0

    # 遍历所有类别对 (A, B)
    for i in range(C):
        class_A = classes[i]
        for j in range(i + 1, C):
            class_B = classes[j]

            # 找到两个类别的共同doy
            doy_A = set(class_doy_means[class_A].keys())
            doy_B = set(class_doy_means[class_B].keys())
            common_doy = doy_A.intersection(doy_B)

            if len(common_doy) == 0:
                continue

            # 计算该类别对的差异
            pair_diff_sum = 0.0
            T = len(common_doy)  # 共同的doy数量

            for doy in common_doy:
                mean_A = class_doy_means[class_A][doy]  # [D]
                mean_B = class_doy_means[class_B][doy]  # [D]

                # ✅ 使用绝对值，避免正负抵消
                diff = np.power(mean_A - mean_B, 2) # [D]
                pair_diff_sum += np.sum(diff)

            # 平均到每个时间步和波段
            D = len(mean_A)  # 波段数量
            pair_avg_diff = pair_diff_sum / (T * D)
            total_pairwise_diff += pair_avg_diff

    # 应用归一化系数
    separability = (2.0 / (C * (C - 1))) * total_pairwise_diff

    return separability
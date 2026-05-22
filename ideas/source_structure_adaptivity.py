import torch


class SourceTargetIntraStrengthAdapter:
    """Conservative v2.6.1 adapter.

    The first v2.6.1 version deliberately uses only target-to-source prototype
    margin as a safety gate. Source reliability and temporal mismatch are
    logged for diagnosis, but they do not change the weight yet.
    """

    def __init__(
        self,
        min_factor=0.75,
        max_factor=1.00,
        ema=0.90,
        high_source_reliability=2.0,
        low_source_reliability=0.8,
        high_target_margin=0.20,
        low_target_margin=0.07,
        high_mismatch_cv=0.25,
        low_mismatch_cv=0.08,
        eps=1e-6,
    ):
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)
        self.ema = float(ema)
        self.high_source_reliability = float(high_source_reliability)
        self.low_source_reliability = float(low_source_reliability)
        self.high_target_margin = float(high_target_margin)
        self.low_target_margin = float(low_target_margin)
        self.high_mismatch_cv = float(high_mismatch_cv)
        self.low_mismatch_cv = float(low_mismatch_cv)
        self.eps = float(eps)
        self.factor_ema = None

    def _pool(self, features):
        if features.ndim != 3:
            raise ValueError(f"Expected features with shape [B, T, D], got {tuple(features.shape)}")
        return features.mean(dim=1)

    def _source_reliability(self, source_features, labels):
        pooled = self._pool(source_features)
        centers = []
        compactness = []
        for class_id in labels.unique(sorted=True):
            mask = labels == class_id
            if int(mask.sum().item()) < 2:
                continue
            feats = pooled[mask]
            center = feats.mean(dim=0)
            centers.append(center)
            compactness.append((feats - center).pow(2).sum(dim=1).sqrt().mean())

        if len(centers) < 2 or not compactness:
            return source_features.new_tensor(1.0)

        centers = torch.stack(centers, dim=0)
        center_distances = torch.cdist(centers, centers, p=2)
        center_distances.fill_diagonal_(float("inf"))
        separability = center_distances.min(dim=1).values.mean()
        compact = torch.stack(compactness).mean()
        return separability / compact.clamp_min(self.eps)

    def _target_margin_ratio(self, source_features, labels, target_features):
        source_pooled = self._pool(source_features)
        target_pooled = self._pool(target_features)
        centers = []
        for class_id in labels.unique(sorted=True):
            mask = labels == class_id
            if int(mask.sum().item()) < 2:
                continue
            centers.append(source_pooled[mask].mean(dim=0))

        if len(centers) < 2 or target_pooled.shape[0] < 1:
            return source_features.new_tensor(0.0)

        centers = torch.stack(centers, dim=0)
        distances = torch.cdist(target_pooled, centers, p=2)
        nearest = distances.topk(k=2, largest=False, dim=1).values
        margin = nearest[:, 1] - nearest[:, 0]
        ratio = margin / nearest[:, 1].clamp_min(self.eps)
        return ratio.mean()

    def _temporal_mismatch_cv(self, source_features, labels, target_features):
        source_prototypes = []
        for class_id in labels.unique(sorted=True):
            mask = labels == class_id
            if int(mask.sum().item()) < 2:
                continue
            source_prototypes.append(source_features[mask].mean(dim=0))

        if not source_prototypes or target_features.shape[0] < 1:
            return source_features.new_tensor(0.0)

        source_curve = torch.stack(source_prototypes, dim=0).mean(dim=0)
        target_curve = target_features.mean(dim=0)
        if source_curve.shape[0] != target_curve.shape[0]:
            min_len = min(source_curve.shape[0], target_curve.shape[0])
            source_curve = source_curve[:min_len]
            target_curve = target_curve[:min_len]
        mismatch = (source_curve - target_curve).pow(2).sum(dim=1).sqrt()
        return mismatch.std(unbiased=False) / mismatch.mean().clamp_min(self.eps)

    def update(self, source_features, labels, target_features):
        with torch.no_grad():
            source_features = source_features.detach()
            target_features = target_features.detach()
            labels = labels.detach()
            source_reliability = self._source_reliability(source_features, labels)
            target_margin = self._target_margin_ratio(source_features, labels, target_features)
            mismatch_cv = self._temporal_mismatch_cv(source_features, labels, target_features)

            raw_factor = source_features.new_tensor(1.0)
            source_flag = 0.0
            target_flag = 0.0
            mismatch_flag = 0.0

            if float(source_reliability.item()) >= self.high_source_reliability:
                source_flag = 1.0
            elif float(source_reliability.item()) <= self.low_source_reliability:
                source_flag = -1.0

            if float(target_margin.item()) >= self.high_target_margin:
                target_flag = 1.0
            elif float(target_margin.item()) <= self.low_target_margin:
                raw_factor = raw_factor - 0.10
                target_flag = -1.0

            if float(mismatch_cv.item()) >= self.high_mismatch_cv:
                mismatch_flag = -1.0
            elif float(mismatch_cv.item()) <= self.low_mismatch_cv:
                mismatch_flag = 1.0

            raw_factor = raw_factor.clamp(self.min_factor, self.max_factor)
            raw_value = float(raw_factor.item())
            if self.factor_ema is None:
                self.factor_ema = raw_value
            else:
                self.factor_ema = self.ema * self.factor_ema + (1.0 - self.ema) * raw_value
            factor = max(self.min_factor, min(self.max_factor, self.factor_ema))

        logs = {
            "source_structure_adaptive_factor": float(factor),
            "source_structure_adaptive_raw_factor": raw_value,
            "source_structure_adaptive_source_reliability": float(source_reliability.item()),
            "source_structure_adaptive_target_margin_ratio": float(target_margin.item()),
            "source_structure_adaptive_temporal_mismatch_cv": float(mismatch_cv.item()),
            "source_structure_adaptive_source_flag": source_flag,
            "source_structure_adaptive_target_flag": target_flag,
            "source_structure_adaptive_mismatch_flag": mismatch_flag,
        }
        return factor, logs

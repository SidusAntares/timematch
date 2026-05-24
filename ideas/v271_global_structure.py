from ideas.v271_decomposition import (
    bounded_residual_energy,
    classwise_curve_variance,
    classwise_dynamics_consistency,
    infer_time_smoothing_bandwidth,
    split_trend_residual,
)


def compute_v271_global_structure_loss(
    ordered_feats,
    ordered_positions,
    labels,
    trend_kernel_size=5,
    trend_smoothing_mode="time",
    trend_bandwidth=0.0,
    trend_kernel="gaussian",
    trend_cohesion_trade_off=1.0,
    trend_dynamics_trade_off=0.05,
    residual_variance_trade_off=0.10,
    residual_energy_trade_off=0.05,
    residual_energy_margin=1.0,
    dynamics_mode="cosine",
):
    trend, residual = split_trend_residual(
        ordered_feats,
        positions=ordered_positions,
        kernel_size=trend_kernel_size,
        mode=trend_smoothing_mode,
        bandwidth=trend_bandwidth,
        kernel=trend_kernel,
    )
    trend_cohesion, trend_class_count = classwise_curve_variance(trend, labels)
    trend_dynamics, dynamics_class_count = classwise_dynamics_consistency(
        trend,
        labels,
        positions=ordered_positions,
        mode=dynamics_mode,
    )
    residual_variance, residual_class_count = classwise_curve_variance(residual, labels)
    residual_energy_loss, residual_energy = bounded_residual_energy(
        residual,
        margin=residual_energy_margin,
    )

    total = (
        float(trend_cohesion_trade_off) * trend_cohesion
        + float(trend_dynamics_trade_off) * trend_dynamics
        + float(residual_variance_trade_off) * residual_variance
        + float(residual_energy_trade_off) * residual_energy_loss
    )
    logs = {
        "v271_global_trend_cohesion_loss": trend_cohesion,
        "v271_global_trend_dynamics_loss": trend_dynamics,
        "v271_global_residual_variance_loss": residual_variance,
        "v271_global_residual_energy_loss": residual_energy_loss,
        "v271_global_residual_energy": residual_energy,
        "v271_global_trend_class_count": float(trend_class_count),
        "v271_global_dynamics_class_count": float(dynamics_class_count),
        "v271_global_residual_class_count": float(residual_class_count),
        "v271_global_trend_kernel_size": float(trend_kernel_size),
        "v271_global_trend_smoothing_mode": str(trend_smoothing_mode),
        "v271_global_trend_bandwidth": float(trend_bandwidth),
        "v271_global_trend_bandwidth_effective": float(trend_bandwidth)
        if float(trend_bandwidth) > 0.0
        else infer_time_smoothing_bandwidth(ordered_positions, kernel_size=trend_kernel_size),
        "v271_global_trend_kernel": str(trend_kernel),
        "v271_global_residual_energy_margin": float(residual_energy_margin),
        "v271_global_trend_cohesion_trade_off": float(trend_cohesion_trade_off),
        "v271_global_trend_dynamics_trade_off": float(trend_dynamics_trade_off),
        "v271_global_residual_variance_trade_off": float(residual_variance_trade_off),
        "v271_global_residual_energy_trade_off": float(residual_energy_trade_off),
    }
    return total, logs

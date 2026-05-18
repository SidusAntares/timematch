"""Lightweight registry helpers for structure view implementations."""

VIEW_COMPUTE_REGISTRY = {}


def register_structure_compute(*names):
    def decorator(fn):
        for name in names:
            VIEW_COMPUTE_REGISTRY[str(name).lower()] = fn
        return fn

    return decorator


def get_structure_compute(name):
    key = str(name).lower()
    if key not in VIEW_COMPUTE_REGISTRY:
        raise KeyError(f"No compute function registered for structure view: {name}")
    return VIEW_COMPUTE_REGISTRY[key]

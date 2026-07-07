"""Training entry point placeholder for v3.2.1 local shift TimeMatch."""


def train_timematch_local_shift(*args, **kwargs):
    """Reserved entry point for structure-guided stage-wise local shift.

    The method is intentionally not wired into ``train.py`` yet.  First active
    implementation should:

    1. load a source structure checkpoint;
    2. build source class-stage references;
    3. partition target temporal features by feature changes;
    4. estimate soft stage alignment;
    5. convert it into local target positions;
    6. reuse the base TimeMatch pseudo-label loss.
    """

    raise NotImplementedError("v3.2.1 local-shift training is a reserved hook, not an active method yet.")

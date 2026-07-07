"""Shift estimation exports for the active TimeMatch baseline.

The implementation currently lives in ``train_loop.py`` for compatibility.
Future refactors should move the function body here and keep this public import
stable.
"""

from methods.timematch_base.train_loop import estimate_temporal_shift, estimate_temporal_shift_details

__all__ = ["estimate_temporal_shift", "estimate_temporal_shift_details"]


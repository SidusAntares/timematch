"""Compatibility wrapper for the active TimeMatch training loop.

New method code should live under ``methods/``.  This module is kept so older
imports such as ``from timematch import train_timematch`` continue to work.
"""

from methods.timematch_base.train_loop import train_timematch

__all__ = ["train_timematch"]

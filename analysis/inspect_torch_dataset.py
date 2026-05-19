#!/usr/bin/env python3
"""Print a compact structure summary for a torch-saved dataset file."""

import argparse
import os
import sys

import torch


def _shape(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return f"len={len(value)}"
    return type(value).__name__


def _summarize(value, indent=0, max_depth=4, max_items=30):
    prefix = "  " * indent
    if indent >= max_depth:
        print(f"{prefix}... max depth reached ({type(value).__name__})")
        return
    if isinstance(value, dict):
        print(f"{prefix}dict keys={list(value.keys())[:max_items]}")
        for idx, (key, child) in enumerate(value.items()):
            if idx >= max_items:
                print(f"{prefix}  ... {len(value) - max_items} more keys")
                break
            print(f"{prefix}  [{repr(key)}] {type(child).__name__} shape={_shape(child)}")
            _summarize(child, indent + 2, max_depth=max_depth, max_items=max_items)
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}{type(value).__name__} len={len(value)}")
        for idx, child in enumerate(value[: min(len(value), max_items)]):
            print(f"{prefix}  [{idx}] {type(child).__name__} shape={_shape(child)}")
            _summarize(child, indent + 2, max_depth=max_depth, max_items=max_items)
    else:
        print(f"{prefix}{type(value).__name__} shape={_shape(value)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a torch .pt/.pth file")
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=30)
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(args.path)
    obj = torch.load(args.path, map_location="cpu", weights_only=False)
    print(f"path: {args.path}")
    print(f"root type: {type(obj).__name__}")
    _summarize(obj, max_depth=args.max_depth, max_items=args.max_items)


if __name__ == "__main__":
    main()

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v28_audit_utils import state_dict_sha256


def main():
    parser = argparse.ArgumentParser(description="Strict-load cleaned source checkpoints with the old DA model class.")
    parser.add_argument("--old_root", required=True)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old_root = str(Path(args.old_root).resolve())
    sys.path.insert(0, old_root)
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    module = importlib.import_module("models.stclassifier")
    import torch

    with Path(args.source_manifest).open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    selected = [
        row for row in manifest
        if row["source_domain"] in {"AT1", "FR2"}
        and row["config"] in {"base", "smooth_k3"}
        and row["seed"] == "1"
    ]
    rows = []
    for row in selected:
        checkpoint_path = Path(row["cleaned_checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        decoder_weights = [
            (name, value)
            for name, value in state.items()
            if name.startswith("decoder.") and name.endswith(".weight") and value.ndim == 2
        ]
        if not decoder_weights:
            raise RuntimeError(f"Cannot infer num_classes from {checkpoint_path}")
        _, final_decoder_weight = min(decoder_weights, key=lambda item: int(item[1].shape[0]))
        model = module.PseLTae(
            input_dim=10,
            num_classes=int(final_decoder_weight.shape[0]),
            with_extra=False,
        )
        incompatible = model.load_state_dict(state, strict=True)
        loaded_hash = state_dict_sha256(model.state_dict())
        source_hash = state_dict_sha256(state)
        rows.append(
            {
                "source_domain": row["source_domain"],
                "config": row["config"],
                "seed": row["seed"],
                "checkpoint_path": str(checkpoint_path),
                "source_state_dict_sha256": source_hash,
                "loaded_state_dict_sha256": loaded_hash,
                "hash_equal": source_hash == loaded_hash,
                "missing_keys": json.dumps(list(incompatible.missing_keys)),
                "unexpected_keys": json.dumps(list(incompatible.unexpected_keys)),
                "strict_load_ok": not incompatible.missing_keys and not incompatible.unexpected_keys,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if not row["strict_load_ok"] or not row["hash_equal"]]
    print(f"OLD_DA_LOAD_SMOKE|rows={len(rows)}|failed={len(failed)}|output={output}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

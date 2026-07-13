import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.v28_audit_utils import checkpoint_hashes


def main():
    parser = argparse.ArgumentParser(description="Print file and state_dict SHA256 for a checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = checkpoint_hashes(args.checkpoint)
    text = json.dumps(result, ensure_ascii=True, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()

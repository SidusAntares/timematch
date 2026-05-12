import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_glob", required=True, help="glob pattern for candidate selection metric JSON files")
    parser.add_argument("--output_json", required=True, help="path to write the winning selection summary")
    args = parser.parse_args()

    candidates = []
    for path in sorted(glob.glob(args.metrics_glob)):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_metrics_path"] = path
        candidates.append(payload)

    if not candidates:
        raise FileNotFoundError(f"No selection metric JSON files matched: {args.metrics_glob}")

    best = max(candidates, key=lambda x: float(x.get("selection_score", float("-inf"))))
    summary = {
        "best_weights_checkpoint": best.get("selected_weights_checkpoint"),
        "best_selection_score": float(best.get("selection_score", 0.0)),
        "best_metrics_path": best.get("_metrics_path"),
        "all_candidates": [
            {
                "weights_checkpoint": item.get("selected_weights_checkpoint"),
                "selection_score": float(item.get("selection_score", 0.0)),
                "selection_score_mode": item.get("selection_score_mode"),
                "selection_legacy_score": float(item.get("selection_legacy_score", 0.0)),
                "selection_temporal_perturbation_score": float(item.get("selection_temporal_perturbation_score", 0.0)),
                "selection_perturbation_score": float(item.get("selection_perturbation_score", 0.0)),
                "selection_perturbation_label_agreement": float(item.get("selection_perturbation_label_agreement", 0.0)),
                "selection_perturbation_prob_consistency": float(item.get("selection_perturbation_prob_consistency", 0.0)),
                "selection_coverage": float(item.get("selection_coverage", 0.0)),
                "selection_mean_confidence": float(item.get("selection_mean_confidence", 0.0)),
                "selection_teacher_student_agreement": float(item.get("selection_teacher_student_agreement", 0.0)),
                "selection_prediction_entropy": float(item.get("selection_prediction_entropy", 0.0)),
                "selection_class_entropy": float(item.get("selection_class_entropy", 0.0)),
                "selection_high_conf_class_entropy": float(item.get("selection_high_conf_class_entropy", 0.0)),
                "selection_source_prior_js": float(item.get("selection_source_prior_js", 0.0)),
                "selection_shift_stability": float(item.get("selection_shift_stability", 0.0)),
                "selection_max_class_fraction": float(item.get("selection_max_class_fraction", 0.0)),
                "metrics_path": item.get("_metrics_path"),
            }
            for item in candidates
        ],
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

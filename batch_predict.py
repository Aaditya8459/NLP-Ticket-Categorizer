import sys
import argparse
import logging

import pandas as pd

from src.config import Config
from src.triage_engine import TriageEngine
from src.utils import setup_logging, load_data, save_json, print_banner


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch predict ticket categories")
    parser.add_argument("--input", "-i", required=True, help="Input CSV/JSON/Excel file")
    parser.add_argument("--output", "-o", default="reports/batch_predictions.json", 
                        help="Output JSON file for results")
    parser.add_argument("--subject-col", default=None, help="Subject column name")
    parser.add_argument("--body-col", default=None, help="Body column name")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    print_banner("BATCH PREDICTION")

    cfg = Config("config.yaml")
    engine = TriageEngine(cfg)

    try:
        engine.load_artifacts()
    except FileNotFoundError as e:
        print(f"❌ Model not found: {e}. Run train.py first.")
        return 1

    subj_col = args.subject_col or cfg.data_subject_col
    body_col = args.body_col or cfg.data_body_col

    df = pd.read_csv(args.input) if args.input.endswith(".csv") else pd.read_excel(args.input)

    if subj_col not in df.columns or body_col not in df.columns:
        raise ValueError(f"Columns not found. Available: {list(df.columns)}")

    subjects = df[subj_col].fillna("").astype(str).tolist()
    bodies = df[body_col].fillna("").astype(str).tolist()

    logging.info("Predicting %d tickets...", len(subjects))
    results = engine.predict_batch(subjects, bodies)

    output = {
        "meta": {
            "input_file": args.input,
            "total_tickets": len(results),
            "auto_assigned": sum(1 for r in results if r.routing == "AUTO-ASSIGNED"),
            "human_review": sum(1 for r in results if r.routing == "NEEDS HUMAN REVIEW"),
            "urgent": sum(1 for r in results if r.priority == "URGENT"),
        },
        "predictions": [r.to_dict() for r in results],
    }

    save_json(output, args.output)

    print("\n📊 BATCH SUMMARY")
    print(f"   Total tickets:     {output['meta']['total_tickets']}")
    print(f"   Auto-assigned:     {output['meta']['auto_assigned']}")
    print(f"   Needs human review: {output['meta']['human_review']}")
    print(f"   Urgent flagged:    {output['meta']['urgent']}")
    print(f"\n✅ Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

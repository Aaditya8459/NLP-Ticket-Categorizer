import sys
import logging

from src.config import Config
from src.triage_engine import TriageEngine
from src.utils import setup_logging, load_data, save_json, print_banner


def main() -> int:
    setup_logging(logging.INFO)

    print_banner("TICKET TRIAGE — TRAINING PIPELINE")

    cfg = Config("config.yaml")

    df = load_data(
        cfg.data_train_path,
        cfg.data_subject_col,
        cfg.data_body_col,
        cfg.data_label_col,
    )

    if len(df) < 20:
        logging.warning("Very small dataset (%d samples). Expect low confidence.", len(df))

    engine = TriageEngine(cfg)
    metrics = engine.train(
        subjects=df["subject"].tolist(),
        bodies=df["body"].tolist(),
        labels=df["category"].tolist(),
    )

    print("\n📊 EVALUATION METRICS")
    print(f"   Accuracy:  {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall:    {metrics['recall']:.3f}")
    print(f"   F1-Score:  {metrics['f1_score']:.3f}")
    print(f"\n   Confusion Matrix:")
    for row, cls in zip(metrics["confusion_matrix"], metrics["classes"]):
        print(f"   {cls:12s} {row}")

    save_json(metrics, f"{cfg.reports_dir}/evaluation.json")

    print("\n✅ Training complete. Artifacts saved to models/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

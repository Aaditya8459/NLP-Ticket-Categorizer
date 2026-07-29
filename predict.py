import sys
import logging

from src.config import Config
from src.triage_engine import TriageEngine
from src.utils import setup_logging, print_banner


def run_demo(engine: TriageEngine) -> None:
    """Run a set of demo tickets."""
    demos = [
        (
            "Double billing on my corporate card",
            "I was charged twice for the enterprise plan this quarter — $2,400 appears twice on my statement. Please reverse the duplicate charge immediately.",
        ),
        (
            "Production database is down",
            "Our primary PostgreSQL instance went down at 2:15 PM. All customer-facing services are offline. This is a critical outage. Need immediate escalation to the SRE team.",
        ),
        (
            "Request for parental leave starting June",
            "I would like to request 12 weeks of parental leave beginning June 1. I have completed all required paperwork and attached the birth certificate. Please confirm approval.",
        ),
        (
            "Suggestion: dark mode for the portal",
            "It would be great if the customer portal had a dark mode option. Many of us work late and the bright white interface is hard on the eyes. Just a friendly suggestion!",
        ),
        (
            "Issue with my account",
            "I am having an issue with my account. Can someone help me? I am not sure what is wrong but something seems off. Please reach out when you can.",
        ),
        (
            "Salary not deposited — bank account issue?",
            "My salary was not deposited today and HR said to check with payroll support. The bank details on file look correct. Can you verify if the transfer was initiated?",
        ),
    ]

    print_banner("DEMO PREDICTIONS")
    for subject, body in demos:
        result = engine.predict_one(subject, body)
        print(result.to_markdown())
        print()


def main() -> int:
    setup_logging(logging.WARNING)  # Quieter for CLI

    print_banner("TICKET TRIAGE — LIVE PREDICTION")

    cfg = Config("config.yaml")
    engine = TriageEngine(cfg)

    try:
        engine.load_artifacts()
    except FileNotFoundError as e:
        print(f"❌ Model artifacts not found: {e}")
        print("   Run 'python train.py' first to train and save the model.")
        return 1

    print("Type 'demo' to see examples, 'quit' to exit.\n")

    while True:
        subject = input("Subject: ").strip()
        if subject.lower() == "quit":
            break
        if subject.lower() == "demo":
            run_demo(engine)
            continue

        body = input("Body:   ").strip()
        result = engine.predict_one(subject, body)

        print("\n" + "─" * 50)
        print(result.to_markdown())
        print("─" * 50 + "\n")

    print("Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

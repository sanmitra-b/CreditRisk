from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import build_applicant_table
from src.data.preprocessor import engineer_application_features
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the applicant-level feature table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path; defaults to artifacts/applicant_features.parquet.",
    )
    parser.add_argument(
        "--skip-installments",
        action="store_true",
        help="Skip the largest history table for a fast smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    output = args.output or settings.artifact_dir / "applicant_features.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = build_applicant_table(
        settings.data_dir,
        include_installments=not args.skip_installments,
    )
    frame = engineer_application_features(frame)
    frame.to_parquet(output, index=False)
    LOGGER.info(
        "Saved %s rows and %s columns to %s",
        len(frame),
        frame.shape[1],
        output,
    )


if __name__ == "__main__":
    main()


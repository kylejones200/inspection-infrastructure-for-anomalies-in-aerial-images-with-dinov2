"""CLI entry point for generating article figures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import signalplot

from infrastructure_dinov2.config import load_config
from infrastructure_dinov2.paths import DEFAULT_CONFIG_PATH
from infrastructure_dinov2.plots import (
    create_anomaly_distribution_visualization,
    create_main_visualization,
    create_performance_metrics_visualization,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate DINOv2 infrastructure inspection visualizations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: repo config.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    _configure_logging(config.logging_level)
    logger = logging.getLogger(__name__)

    signalplot.apply(font_family=config.font_family)

    logger.info("Infrastructure inspection — DINOv2 visualizations")
    paths = [
        create_main_visualization(config),
        create_anomaly_distribution_visualization(config),
        create_performance_metrics_visualization(config),
    ]
    saved = [p for p in paths if p is not None]

    if saved:
        logger.info("Wrote %s figure(s) to %s", len(saved), config.output.figures_dir)
        for path in saved:
            logger.info("  - %s", path.name)
    else:
        logger.info("No figures written (save_figures=false)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

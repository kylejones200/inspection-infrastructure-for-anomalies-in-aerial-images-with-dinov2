"""Load and validate project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infrastructure_dinov2.paths import DEFAULT_CONFIG_PATH, resolve_project_path


@dataclass(frozen=True)
class TsneConfig:
    perplexity: int
    max_iter: int


@dataclass(frozen=True)
class DataConfig:
    n_images: int
    embedding_dim: int
    seed: int
    class_fractions: dict[str, float]


@dataclass(frozen=True)
class OutputConfig:
    figures_dir: Path
    save_figures: bool
    figure_dpi: int
    main_figure: str
    distribution_figure: str
    performance_figure: str


@dataclass(frozen=True)
class AppConfig:
    logging_level: str
    data: DataConfig
    tsne: TsneConfig
    anomaly_threshold_sigma: float
    output: OutputConfig
    font_family: str


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    data_raw = _require(raw, "data")
    viz_raw = _require(raw, "visualization")
    output_raw = _require(raw, "output")
    style_raw = raw.get("style", {})
    return AppConfig(
        logging_level=raw.get("logging", {}).get("level", "INFO"),
        data=DataConfig(
            n_images=int(_require(data_raw, "n_images")),
            embedding_dim=int(_require(data_raw, "embedding_dim")),
            seed=int(_require(data_raw, "seed")),
            class_fractions=dict(_require(data_raw, "class_fractions")),
        ),
        tsne=TsneConfig(
            perplexity=int(_require(viz_raw.get("tsne", {}), "perplexity")),
            max_iter=int(_require(viz_raw.get("tsne", {}), "max_iter")),
        ),
        anomaly_threshold_sigma=float(_require(viz_raw, "anomaly_threshold_sigma")),
        output=OutputConfig(
            figures_dir=resolve_project_path(_require(output_raw, "figures_dir")),
            save_figures=bool(output_raw.get("save_figures", True)),
            figure_dpi=int(output_raw.get("figure_dpi", 300)),
            main_figure=str(_require(output_raw, "main_figure")),
            distribution_figure=str(_require(output_raw, "distribution_figure")),
            performance_figure=str(_require(output_raw, "performance_figure")),
        ),
        font_family=str(style_raw.get("font_family", "serif")),
    )

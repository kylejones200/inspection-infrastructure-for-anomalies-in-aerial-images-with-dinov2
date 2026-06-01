"""Tests for configuration loading."""

from infrastructure_dinov2.config import load_config
from infrastructure_dinov2.paths import DEFAULT_CONFIG_PATH, FIGURES_DIR


def test_load_default_config():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.data.n_images == 10_000
    assert config.data.embedding_dim == 384
    assert config.output.figures_dir == FIGURES_DIR
    assert config.output.save_figures is True

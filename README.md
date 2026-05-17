# Inspection Infrastructure for Anomalies in Aerial Images with DINOv2

Published: 2025-11-06  
Medium: [Inspection Infrastructure for Anomalies in Aerial Images with DINOv2](https://medium.com/@kyle-t-jones/inspection-infrastructure-for-anomalies-in-aerial-images-with-dinov2-362dfccd288d)

Companion code for the article (`article.md`). Generates synthetic DINOv2-style embedding visualizations (t-SNE map, anomaly score distribution, and review-workload metrics) that illustrate zero-shot anomaly detection at infrastructure scale.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run infra-dinov2-viz
```

Figures are written to `outputs/figures/`:

| File | Description |
|------|-------------|
| `infrastructure_embeddings_tsne.png` | t-SNE of 384-dim embeddings with flagged outliers |
| `anomaly_score_distribution.png` | Score histogram with μ + 3σ threshold |
| `review_workload_metrics.png` | Review workload vs. detection performance |

## Project layout

```
config.yaml                 # sample size, t-SNE, output paths
pyproject.toml / uv.lock
src/infrastructure_dinov2/  # embeddings, plots, CLI
tests/
outputs/figures/            # generated PNGs (gitignored except .gitkeep)
docs/blog.md                # extended article draft with figure references
article.md                  # Medium export
```

## Configuration

Edit `config.yaml` to change `data.n_images`, class fractions, t-SNE settings, or output filenames. Set `output.save_figures: false` to run the pipeline without writing PNGs.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

CI runs ruff and pytest on push/PR (see `.github/workflows/ci.yml`).

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).

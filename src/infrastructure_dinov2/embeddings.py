"""Synthetic DINOv2-style embeddings for demonstration visualizations."""

from __future__ import annotations

import numpy as np

from infrastructure_dinov2.config import DataConfig

LABEL_NORMAL = 0
LABEL_VEGETATION = 1
LABEL_EQUIPMENT = 2
LABEL_DAMAGE = 3

CLASS_CENTERS = {
    LABEL_NORMAL: np.zeros(384),
    LABEL_VEGETATION: np.ones(384) * 0.5,
    LABEL_EQUIPMENT: np.ones(384) * 1.5,
    LABEL_DAMAGE: np.ones(384) * -1.0,
}

CLASS_SCALES = {
    LABEL_NORMAL: 0.3,
    LABEL_VEGETATION: 0.4,
    LABEL_EQUIPMENT: 0.8,
    LABEL_DAMAGE: 0.6,
}


def class_counts(n_images: int, fractions: dict[str, float]) -> tuple[int, int, int, int]:
    """Return counts for normal, vegetation, equipment, and damage classes."""
    n_normal = int(n_images * fractions["normal"])
    n_vegetation = int(n_images * fractions["vegetation"])
    n_equipment = int(n_images * fractions["equipment"])
    n_damage = n_images - n_normal - n_vegetation - n_equipment
    return n_normal, n_vegetation, n_equipment, n_damage


def generate_embeddings_with_structure(
    data: DataConfig | None = None,
    *,
    n_images: int | None = None,
    embedding_dim: int | None = None,
    seed: int | None = None,
    class_fractions: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic embeddings with realistic cluster structure.
    Simulates DINOv2 embeddings:
    - ~94% normal infrastructure (tight cluster)
    - ~3% vegetation intrusion
    - ~2% equipment/activity
    - remainder surface damage
    Returns:
        embeddings: (n_images, embedding_dim)
        labels: int array (0=normal, 1=vegetation, 2=equipment, 3=damage)
        anomaly_scores: distance to each point's class centroid
    """
    if data is not None:
        n_images = data.n_images
        embedding_dim = data.embedding_dim
        seed = data.seed
        class_fractions = data.class_fractions
    if n_images is None or embedding_dim is None or seed is None or class_fractions is None:
        raise ValueError("Provide DataConfig or all keyword arguments")

    rng = np.random.default_rng(seed)
    counts = class_counts(n_images, class_fractions)
    embeddings_parts: list[np.ndarray] = []
    labels: list[int] = []
    for label_id, count in enumerate(counts):
        center = _center_for_dim(CLASS_CENTERS[label_id], embedding_dim)
        scale = CLASS_SCALES[label_id]
        part = rng.standard_normal((count, embedding_dim)) * scale + center
        embeddings_parts.append(part)
        labels.extend([label_id] * count)

    embeddings = np.vstack(embeddings_parts)
    labels_arr = np.array(labels, dtype=np.int64)
    centers = {label_id: _center_for_dim(c, embedding_dim) for label_id, c in CLASS_CENTERS.items()}
    anomaly_scores = np.array(
        [np.linalg.norm(embeddings[i] - centers[int(label)]) for i, label in enumerate(labels_arr)]
    )
    return embeddings, labels_arr, anomaly_scores


def _center_for_dim(center: np.ndarray, embedding_dim: int) -> np.ndarray:
    if center.shape[0] == embedding_dim:
        return center
    if center.shape[0] == 384 and embedding_dim != 384:
        return np.resize(center, embedding_dim)
    raise ValueError(f"Unsupported embedding_dim={embedding_dim}")

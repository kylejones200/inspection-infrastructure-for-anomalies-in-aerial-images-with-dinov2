"""Tests for synthetic embedding generation."""

import numpy as np

from infrastructure_dinov2.config import DataConfig
from infrastructure_dinov2.embeddings import (
    LABEL_DAMAGE,
    LABEL_EQUIPMENT,
    LABEL_NORMAL,
    LABEL_VEGETATION,
    class_counts,
    generate_embeddings_with_structure,
)


def test_class_counts_sum_to_total():
    n = 10_000
    fractions = {"normal": 0.94, "vegetation": 0.03, "equipment": 0.02}
    counts = class_counts(n, fractions)
    assert sum(counts) == n
    assert counts[3] >= 0


def test_generate_embeddings_shape_and_labels():
    data = DataConfig(
        n_images=500,
        embedding_dim=64,
        seed=7,
        class_fractions={"normal": 0.94, "vegetation": 0.03, "equipment": 0.02},
    )
    embeddings, labels, scores = generate_embeddings_with_structure(data)
    assert embeddings.shape == (500, 64)
    assert labels.shape == (500,)
    assert scores.shape == (500,)
    assert set(labels.tolist()) <= {LABEL_NORMAL, LABEL_VEGETATION, LABEL_EQUIPMENT, LABEL_DAMAGE}


def test_generate_embeddings_reproducible():
    kwargs = dict(
        n_images=100,
        embedding_dim=32,
        seed=99,
        class_fractions={"normal": 0.9, "vegetation": 0.05, "equipment": 0.03},
    )
    e1, l1, s1 = generate_embeddings_with_structure(**kwargs)
    e2, l2, s2 = generate_embeddings_with_structure(**kwargs)
    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_array_equal(s1, s2)

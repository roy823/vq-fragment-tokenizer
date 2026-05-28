import numpy as np

from vqfrag.metrics import code_usage_stats, peak_recall_ppm, spectral_cosine


def test_spectral_cosine_and_peak_recall():
    spec_a = [(100.0, 1.0), (200.0, 0.5)]
    spec_b = [(100.0, 0.8), (300.0, 0.1)]

    assert 0 < spectral_cosine(spec_a, spec_b, bin_width=0.1) < 1
    assert peak_recall_ppm([100.0, 200.0], [100.001], ppm=20) == 0.5


def test_code_usage_stats():
    stats = code_usage_stats(np.array([0, 0, 1, 3]), codebook_size=4)

    assert stats["active_codes"] == 3
    assert stats["perplexity_fraction"] > 0

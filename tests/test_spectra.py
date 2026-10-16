import numpy as np
import pytest

from casmi.spectra import SearchConfig, cosine, prepare


def test_cleaning_filters_noise_caps_peaks_and_normalizes():
    mz, weights = prepare(
        [40, 10, 20, 30, 103, np.nan], [0.00001, 0.5, 1, 0.25, 1, 1],
        100, SearchConfig(max_peaks=2),
    )
    np.testing.assert_array_equal(mz, [10, 20])
    assert np.linalg.norm(weights) == pytest.approx(1)
    assert weights[1] / weights[0] == pytest.approx(np.sqrt(2))


def test_cosine_matches_each_peak_at_most_once():
    score, count = cosine(
        np.array([10.0, 10.01]), np.array([0.6, 0.8]),
        np.array([10.005]), np.array([1.0]), 0.02,
    )
    assert count == 1
    assert score == pytest.approx(0.8)


def test_cosine_respects_fragment_mass_tolerance():
    mz = np.array([10.0, 20.0, 30.0])
    weights = np.ones(3) / np.sqrt(3)
    assert cosine(mz, weights, mz + 0.01, weights, 0.02)[0] == pytest.approx(1)
    assert cosine(mz, weights, mz + 0.1, weights, 0.02) == (0, 0)

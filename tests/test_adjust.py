import pytest

from mercato.features.adjust import (
    ewma_form,
    form_lambda,
    inflate_fee,
    match_weight,
    padj,
    per90,
    shrink,
)


def test_per90():
    assert per90(12, 1740) == pytest.approx(0.6207, abs=1e-3)
    assert per90(3, 0) == 0.0


def test_shrink_small_sample_leans_to_average():
    # One goal in 200 minutes is 0.45 per 90, avg 0.30
    s = shrink(0.45, 200, 0.30, k=900)
    assert 0.30 < s < 0.35


def test_shrink_big_sample_stays_near_player():
    assert shrink(0.6, 3000, 0.3, k=900) == pytest.approx(0.531, abs=1e-3)


def test_padj_matches_deck_examples():
    assert padj(4.0, 0.40) == pytest.approx(3.33, abs=0.01)
    assert padj(4.0, 0.65) == pytest.approx(5.71, abs=0.01)


def test_match_weight_orders_opponents():
    assert match_weight(2000) > match_weight(1700) == 1.0 > match_weight(1550)


def test_form_half_life():
    lam = form_lambda(5)
    assert (1 - lam) ** 5 == pytest.approx(0.5)
    assert ewma_form([0, 0, 3, 3]) > ewma_form([3, 3, 0, 0])
    assert ewma_form([3, 3, 3]) == pytest.approx(3.0)


def test_inflate_fee():
    assert inflate_fee(40e6, index_year=10e6, index_now=20e6) == 80e6

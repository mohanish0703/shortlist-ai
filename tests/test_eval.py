"""Ranking metrics, the keyword baseline and the fairness swap; no API calls."""

from __future__ import annotations

import itertools
import statistics

import pytest

from shortlist.baselines import bm25
from shortlist.eval_fairness import swap
from shortlist.eval_rank import chance, ndcg, precision_at


def test_ndcg_is_one_for_the_ideal_order_and_lower_otherwise():
    ideal = ["good", "good", "borderline", "poor"]
    assert ndcg(ideal) == pytest.approx(1.0)
    # gains 3, 3, 1, 0 (2^g - 1); reversed: (0 + 1/log2 3 + 3/2 + 3/log2 5) / (3 + 3/log2 3 + 1/2)
    assert ndcg(list(reversed(ideal))) == pytest.approx(0.6352, abs=1e-3)
    assert ndcg(["poor", "poor"]) == 0.0  # nothing relevant: defined as 0


def test_precision_at_5():
    assert precision_at(["good", "poor", "good", "good", "borderline", "good"]) == 0.6


def test_bm25_prefers_matching_text():
    docs = {"a": "housekeeping staff in bangalore", "b": "delivery rider in chennai"}
    scores = bm25("housekeeping bangalore", docs)
    assert scores["a"] > scores["b"] == 0


def test_swap_keeps_upper_case():
    text = "NAME: ASHA RAO\nMy name is Asha Rao."
    assert swap(text, "Asha Rao", "Salim Khan") == "NAME: SALIM KHAN\nMy name is Salim Khan."


def test_chance_matches_the_average_over_every_order():
    ls = ["good", "good", "borderline", "poor", "poor", "good"]
    orders = [list(p) for p in itertools.permutations(ls)]
    assert chance(ls)["ndcg@10"] == pytest.approx(statistics.mean(map(ndcg, orders)))
    assert chance(ls)["precision@5"] == pytest.approx(statistics.mean(map(precision_at, orders)))

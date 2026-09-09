from eval.util.phase3 import aggregate_eval105_summaries


def _summary(seed, clean, combined):
    return {
        "seed": seed,
        "seed_type": "paired_replay_perturbation_seed",
        "conditions": [
            {"condition": "clean", "accuracy_percent": clean, "paired_accuracy_delta_percent_vs_clean": 0.0},
            {"condition": "combined", "accuracy_percent": combined, "paired_accuracy_delta_percent_vs_clean": combined - clean},
        ],
    }


def test_accuracy_aggregation_reports_mean_std_and_incomplete_status():
    rows, meta = aggregate_eval105_summaries(
        [_summary(0, 80.0, 76.0), _summary(1, 82.0, 77.0)],
        minimum_seeds=5,
    )
    combined = next(row for row in rows if row["condition"] == "combined")
    assert combined["accuracy_mean_percent"] == 76.5
    assert combined["accuracy_std_percent"] > 0
    assert combined["statistics_status"] == "insufficient_replay_seeds"
    assert meta["seed_scope"].startswith("paired_replay_perturbation_seed")


def test_accuracy_aggregation_marks_sufficient_replay_set():
    rows, meta = aggregate_eval105_summaries(
        [_summary(seed, 80.0 + seed, 76.0 + seed) for seed in range(5)],
        minimum_seeds=5,
    )
    assert meta["sufficient_seed_count"]
    assert all(row["statistics_status"] == "complete" for row in rows)

"""
Automated Pytest Suite: Scientific Benchmark Evaluation.
Tests benchmark dataset integrity, multi-mode evaluation, metric computation, and CSV export.
"""

import os
import pytest
from src.evaluation.benchmark import load_benchmark_cases
from src.evaluation.evaluator import VerificationEvaluator


def test_benchmark_dataset_integrity():
    cases = load_benchmark_cases()
    assert len(cases) >= 5

    case_ids = [c["id"] for c in cases]
    assert "CASE-01-BELL-VALID" in case_ids
    assert "CASE-02-BELL-INVALID-DIST" in case_ids
    assert "CASE-03-INVALID-QUBIT-GATE" in case_ids
    assert "CASE-04-INCOMPLETE-PROPOSAL" in case_ids

    for c in cases:
        assert "ground_truth_label" in c
        assert "expected_overall_status" in c
        assert "proposal" in c


def test_evaluator_execution_and_metrics(tmp_path):
    csv_file = str(tmp_path / "test_evaluation_results.csv")
    evaluator = VerificationEvaluator(output_csv=csv_file)

    df, summary = evaluator.run_evaluation()

    assert os.path.exists(csv_file)
    assert len(df) == len(evaluator.cases) * 3  # 3 modes tested

    # Check that summary metrics exist for each mode
    assert "Rule & Simulation Only" in summary
    assert "Hybrid (Rules + Sim + RAG)" in summary

    hybrid_metrics = summary["Hybrid (Rules + Sim + RAG)"]
    assert hybrid_metrics["precision"] >= 0.90
    assert hybrid_metrics["recall"] >= 0.90
    assert hybrid_metrics["false_pass_rate"] == 0.0  # Zero invalid proposals should falsely pass
    assert hybrid_metrics["simulation_agreement"] == 1.0

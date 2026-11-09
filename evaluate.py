"""
Independent Evaluation Runner for Quantum Experiment Verification System.
Executes the reproducible benchmark suite, calculates metrics, and exports CSV.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.evaluation.evaluator import VerificationEvaluator


def main():
    print("=" * 80)
    print("AI-BASED VERIFICATION OF QUANTUM EXPERIMENT PROPOSALS - BENCHMARK EVALUATION")
    print("=" * 80)

    csv_path = "evaluation_results.csv"
    evaluator = VerificationEvaluator(output_csv=csv_path)

    print(f"Loaded {len(evaluator.cases)} labelled benchmark test cases.")
    print("Executing multi-tier evaluation...\n")

    df, summary = evaluator.run_evaluation()

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION RESULTS (AGGREGATED BY MODE)")
    print("=" * 80)
    
    header = f"{'Verification Mode':<30} | {'Prec.':<7} | {'Recall':<7} | {'FPR':<7} | {'FAR':<7} | {'Sim Agree':<10} | {'Latency (ms)':<12}"
    print(header)
    print("-" * len(header))

    for mode, m in summary.items():
        print(
            f"{mode:<30} | "
            f"{m['precision']:<7.4f} | "
            f"{m['recall']:<7.4f} | "
            f"{m['false_pass_rate']:<7.4f} | "
            f"{m['false_alarm_rate']:<7.4f} | "
            f"{m['simulation_agreement']:<10.4f} | "
            f"{m['avg_execution_time_ms']:<12.2f}"
        )

    print("\n" + "=" * 80)
    print("PER-CASE BREAKDOWN (HYBRID MODE)")
    print("=" * 80)
    hybrid_df = df[df["mode"] == "Hybrid (Rules + Sim + RAG)"]
    for _, row in hybrid_df.iterrows():
        status_flag = "PASS" if row["is_correct"] else "FAIL"
        print(
            f"[{row['case_id']}] {row['case_name'][:45]:<45} | "
            f"Expected: {row['expected_status']:<12} | "
            f"Actual: {row['actual_status']:<12} | "
            f"Agreement: {status_flag}"
        )

    print(f"\nDetailed evaluation metrics saved to: {os.path.abspath(csv_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

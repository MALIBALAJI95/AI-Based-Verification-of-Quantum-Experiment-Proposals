"""
Reproducible Scientific Evaluation Harness.
Measures Error Detection Precision, Recall, False-Pass Rate, False-Alarm Rate,
Simulation Agreement, Extraction Completeness, and Execution Latency.
Compares verification modes and exports structured results to CSV.
"""

import time
import os
import csv
from typing import List, Dict, Any, Tuple
import pandas as pd

from ..schemas.proposal import ProposalSpecification
from ..schemas.verification import CheckStatus, VerificationReport
from ..verification.engine import VerificationEngine
from ..knowledge_base.vector_store import QuantumKnowledgeBase
from ..knowledge_base.retriever import QuantumRetriever
from ..llm.client import ConfigurableLLMClient
from .benchmark import load_benchmark_cases


class VerificationEvaluator:
    """
    Evaluates verification engine against labelled benchmark proposals across multiple configurations.
    """

    def __init__(self, output_csv: str = "evaluation_results.csv"):
        self.output_csv = output_csv
        self.cases = load_benchmark_cases()
        
        # Initialize knowledge base and retriever
        self.kb = QuantumKnowledgeBase()
        self.kb.seed_starter_corpus()
        self.retriever = QuantumRetriever(self.kb)

    def run_evaluation(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute benchmark evaluation across three distinct verification modalities:
        1. Rule-based and simulation checks (no LLM, no RAG)
        2. LLM analysis without RAG
        3. Full Hybrid Verification (Rules + Simulation + RAG + LLM)
        """
        records: List[Dict[str, Any]] = []

        modes = [
            {"mode": "Rule & Simulation Only", "use_rag": False, "use_llm": False},
            {"mode": "LLM Analysis (No RAG)", "use_rag": False, "use_llm": True},
            {"mode": "Hybrid (Rules + Sim + RAG)", "use_rag": True, "use_llm": True},
        ]

        for m_cfg in modes:
            mode_name = m_cfg["mode"]
            use_rag = m_cfg["use_rag"]
            use_llm = m_cfg["use_llm"]

            retriever_inst = self.retriever if use_rag else None
            llm_client = ConfigurableLLMClient(demo_mode=True) if use_llm else None
            engine = VerificationEngine(retriever=retriever_inst, llm_client=llm_client)

            for case in self.cases:
                cid = case["id"]
                cname = case["name"]
                ground_truth = case["ground_truth_label"]  # VALID, INVALID, INCOMPLETE
                expected_status = case["expected_overall_status"] # PASS, FAIL, INCONCLUSIVE
                prop_data = case["proposal"]
                
                proposal = ProposalSpecification(**prop_data)

                start_t = time.perf_counter()
                report = engine.verify_proposal(proposal)
                exec_ms = (time.perf_counter() - start_t) * 1000.0

                actual_status = self._determine_overall_status(report)

                # Correctness determination
                is_correct = (actual_status == expected_status)

                # Extraction completeness
                has_q = proposal.num_qubits is not None
                has_c = proposal.circuit is not None and len(proposal.circuit.gates) > 0
                has_o = len(proposal.expected_outcomes) > 0
                completeness = sum([has_q, has_c, has_o]) / 3.0

                # Simulation agreement with expected ground truth status
                sim_agree = 1.0 if is_correct else 0.0

                # Citations count
                num_cits = len(report.citations) if use_rag else 0

                records.append({
                    "case_id": cid,
                    "case_name": cname,
                    "ground_truth": ground_truth,
                    "expected_status": expected_status,
                    "mode": mode_name,
                    "actual_status": actual_status,
                    "is_correct": is_correct,
                    "checks_passed": report.summary_counts.get("PASS", 0),
                    "checks_failed": report.summary_counts.get("FAIL", 0),
                    "checks_inconclusive": report.summary_counts.get("INCONCLUSIVE", 0),
                    "extraction_completeness": round(completeness, 2),
                    "simulation_agreement": sim_agree,
                    "citations_retrieved": num_cits,
                    "execution_time_ms": round(exec_ms, 2)
                })

        df = pd.DataFrame(records)
        df.to_csv(self.output_csv, index=False)

        # Aggregate summary metrics per mode
        summary_metrics = self._compute_summary_metrics(df)
        return df, summary_metrics

    def _determine_overall_status(self, report: VerificationReport) -> str:
        if report.summary_counts.get("FAIL", 0) > 0:
            return "FAIL"
        if report.summary_counts.get("INCONCLUSIVE", 0) > 0:
            return "INCONCLUSIVE"
        return "PASS"

    def _compute_summary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute precision, recall, false-pass rate, and false-alarm rate per mode."""
        summary = {}

        for mode, group in df.groupby("mode"):
            # Ground truth invalid includes INVALID and INCOMPLETE (should not pass)
            total_cases = len(group)
            
            # True positives: ground truth is INVALID/INCOMPLETE and system detected it (FAIL or INCONCLUSIVE)
            # False pass: ground truth is INVALID/INCOMPLETE but system reported PASS
            # False alarm: ground truth is VALID but system reported FAIL

            invalid_cases = group[group["ground_truth"].isin(["INVALID", "INCOMPLETE"])]
            valid_cases = group[group["ground_truth"] == "VALID"]

            # False pass: invalid marked as PASS
            false_passes = len(invalid_cases[invalid_cases["actual_status"] == "PASS"])
            false_pass_rate = (false_passes / len(invalid_cases)) if len(invalid_cases) > 0 else 0.0

            # False alarm: valid marked as FAIL
            false_alarms = len(valid_cases[valid_cases["actual_status"] == "FAIL"])
            false_alarm_rate = (false_alarms / len(valid_cases)) if len(valid_cases) > 0 else 0.0

            # Error detection: Detecting INVALID cases as FAIL
            strictly_invalid = group[group["ground_truth"] == "INVALID"]
            true_errors_detected = len(strictly_invalid[strictly_invalid["actual_status"] == "FAIL"])
            recall = (true_errors_detected / len(strictly_invalid)) if len(strictly_invalid) > 0 else 1.0

            all_fails = group[group["actual_status"] == "FAIL"]
            precision = (true_errors_detected / len(all_fails)) if len(all_fails) > 0 else 1.0

            avg_time = group["execution_time_ms"].mean()
            sim_agreement = group["simulation_agreement"].mean()

            summary[mode] = {
                "total_cases": total_cases,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "false_pass_rate": round(false_pass_rate, 4),
                "false_alarm_rate": round(false_alarm_rate, 4),
                "simulation_agreement": round(sim_agreement, 4),
                "avg_execution_time_ms": round(avg_time, 2),
            }

        return summary

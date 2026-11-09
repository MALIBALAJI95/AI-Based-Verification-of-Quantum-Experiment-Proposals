"""
Benchmark Dataset Loader for Quantum Experiment Verification.
Loads labelled test proposals (valid, invalid distributions, syntax errors, incomplete specs).
"""

import os
import json
import glob
from typing import List, Dict, Any

BENCHMARK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "benchmark_cases"))


def load_benchmark_cases() -> List[Dict[str, Any]]:
    """Load all JSON test cases from the benchmark directory."""
    files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "*.json")))
    cases = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            cases.append(json.load(f))
    return cases

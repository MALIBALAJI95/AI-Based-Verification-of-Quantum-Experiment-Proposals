"""
Quantum verification metrics: Phase-invariant state fidelity, Total Variation Distance (TVD),
and expected-vs-computed probability distribution comparisons.
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..schemas.proposal import ExpectedOutcome
from ..schemas.verification import OutcomeComparison


def total_variation_distance(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    """
    Compute Total Variation Distance (TVD) between two discrete probability distributions:
    TVD(P, Q) = 0.5 * sum_x |P(x) - Q(x)|.
    Bounded in [0.0, 1.0].
    """
    all_keys = set(dist_a.keys()).union(set(dist_b.keys()))
    tvd = 0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in all_keys)
    return float(tvd)


def compute_fidelity(sv_a: np.ndarray, sv_b: np.ndarray) -> float:
    """
    Compute phase-invariant quantum state fidelity between two pure statevectors:
    F(|psi>, |phi>) = |<psi|phi>|^2.
    Bounded in [0.0, 1.0].
    """
    if len(sv_a) != len(sv_b):
        raise ValueError(f"Statevector dimension mismatch: {len(sv_a)} vs {len(sv_b)}")
    
    # Normalize vectors just in case
    norm_a = np.linalg.norm(sv_a)
    norm_b = np.linalg.norm(sv_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    v_a = sv_a / norm_a
    v_b = sv_b / norm_b

    inner_prod = np.vdot(v_a, v_b)
    fid = float(abs(inner_prod) ** 2)
    # Clip numerical precision artifacts
    return max(0.0, min(1.0, fid))


def compare_outcomes(
    expected_outcomes: List[ExpectedOutcome],
    computed_probabilities: Dict[str, float],
    default_tolerance: float = 0.05
) -> Tuple[List[OutcomeComparison], bool, float]:
    """
    Compare author-specified expected outcome probabilities with computed simulation probabilities.
    Returns:
      - List of OutcomeComparison records
      - all_matched (bool): True iff all expected outcomes are within tolerance
      - max_error (float): Maximum discrepancy observed
    """
    comparisons: List[OutcomeComparison] = []
    all_matched = True
    max_error = 0.0

    # Ensure all bitstrings have consistent length formatting
    for exp in expected_outcomes:
        basis = exp.basis_state.strip()
        expected_p = exp.expected_probability
        computed_p = computed_probabilities.get(basis, 0.0)
        
        diff = abs(expected_p - computed_p)
        tolerance = exp.tolerance if exp.tolerance is not None else default_tolerance
        is_match = diff <= (tolerance + 1e-9)

        if not is_match:
            all_matched = False

        if diff > max_error:
            max_error = diff

        comparisons.append(
            OutcomeComparison(
                basis_state=basis,
                expected_probability=round(expected_p, 4),
                computed_probability=round(computed_p, 4),
                absolute_difference=round(diff, 4),
                tolerance=round(tolerance, 4),
                is_match=is_match
            )
        )

    return comparisons, all_matched, max_error

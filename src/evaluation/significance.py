"""
significance.py — Statistical significance testing via paired bootstrap resampling and paired t-tests.
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np


def paired_bootstrap_test(
    baseline_values: List[float],
    experimental_values: List[float],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes paired bootstrap resampling for metric difference (Experimental - Baseline).
    Returns mean delta, 95% confidence interval, and empirical two-tailed p-value.
    """
    assert len(baseline_values) == len(experimental_values), "Arrays must be paired and have identical length."
    n = len(baseline_values)
    if n == 0:
        return {"mean_difference": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False}

    b_arr = np.array(baseline_values)
    e_arr = np.array(experimental_values)
    differences = e_arr - b_arr
    observed_mean_diff = float(np.mean(differences))

    rng = np.random.default_rng(seed)
    bootstrap_diffs = np.empty(num_bootstraps)

    for i in range(num_bootstraps):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_diffs[i] = np.mean(differences[sample_indices])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_diffs, (alpha / 2.0) * 100.0))
    ci_upper = float(np.percentile(bootstrap_diffs, (1.0 - alpha / 2.0) * 100.0))

    # Empirical two-tailed p-value
    if observed_mean_diff > 0:
        p_val = float(2.0 * np.mean(bootstrap_diffs <= 0))
    else:
        p_val = float(2.0 * np.mean(bootstrap_diffs >= 0))
    p_val = min(1.0, max(0.0, p_val))

    # Paired t-test
    mean_d = float(np.mean(differences))
    std_d = float(np.std(differences, ddof=1)) if n > 1 else 0.0
    se_d = (std_d / math.sqrt(n)) if n > 0 else 1.0
    t_stat = (mean_d / se_d) if se_d > 0 else 0.0

    return {
        "n_samples": n,
        "observed_mean_difference": round(observed_mean_diff, 5),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
        "confidence_level": confidence_level,
        "bootstrap_p_value": round(p_val, 5),
        "t_statistic": round(t_stat, 4),
        "is_statistically_significant": bool(p_val < 0.05),
    }


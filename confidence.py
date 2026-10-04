"""confidence.py -- confidence system (section 24).

confidence = f(sample_count, consistency, improvement, variance, age)

Bands:  0.00-0.20 useless | 0.20-0.50 weak | 0.50-0.75 experimental
        0.75-0.90 strong  | 0.90-1.00 highly validated

IMPORTANT: this confidence measures evidence that a *search heuristic* is
associated with productive loops.  It is NOT a probability of solving a
hash target and must never be presented as one (section 24, section 33).
"""

import math

__all__ = ["compute_confidence", "band_label"]


def compute_confidence(sample_count, consistency, effect, variance, age, cfg):
    n_term = 1.0 - math.exp(-sample_count / max(1.0,
                                                float(cfg.min_pattern_samples)))
    # Zero/negative effect => zero confidence, always: consistency or age
    # cannot rescue a pattern that shows no measured improvement.
    if effect <= 1.0:
        return 0.0
    eff = max(0.0, min(1.0, effect - 1.0))
    cons = max(0.0, min(1.0, consistency))
    var_term = 1.0 / (1.0 + max(0.0, variance))
    age_term = 1.0 / (1.0 + 0.02 * max(0, age))
    c = n_term * (0.45 * eff + 0.35 * cons + 0.10 * var_term + 0.10 * age_term)
    return max(0.0, min(1.0, c))


def band_label(c):
    if c < 0.20:
        return "useless"
    if c < 0.50:
        return "weak"
    if c < 0.75:
        return "experimental"
    if c < 0.90:
        return "strong"
    return "highly_validated"

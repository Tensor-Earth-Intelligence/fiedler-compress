"""Per-face metric selection.

For v0.1, the _select_metric() function in faces.py handles metric
selection with simple decision rules. This module is a placeholder
for future learned metric selection (e.g., train a classifier on
benchmark data to predict which metric produces the best compression
quality per face).
"""

from __future__ import annotations

# Future: LearnedMetricSelector class that trains on benchmark data
# and predicts optimal metric per face based on geometric properties.

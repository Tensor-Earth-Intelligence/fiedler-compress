"""
Dymaxion Polyhedral Compression.

Two variants:
- Dymaxion-Fiedler: polyhedral similarity graph + Fiedler spectral decomposition
- Dymaxion-Pure: polyhedral faces + information-theoretic scoring (no spectral)
"""

from fiedler_optimizer.dymaxion.compress import dymaxion_compress

__all__ = ["dymaxion_compress"]

"""Commercial-tier feature gating for the open-core distribution.

Some capabilities are not included in the open-core package. The modules that
implement them are absent here, so a lazy ``import`` of one raises
:class:`ImportError`. The helpers below translate that into a clear,
actionable message instead of an opaque import crash.
"""

from __future__ import annotations


class CommercialTierError(RuntimeError):
    """Raised when an open-core caller requests a commercial-tier capability."""


def commercial_tier_error(feature: str | None = None) -> "CommercialTierError":
    """Return a generic :class:`CommercialTierError` for a gated capability.

    The message intentionally does not name the specific capability.

    Parameters
    ----------
    feature : str, optional
        Accepted for backward compatibility; ignored.
    """
    return CommercialTierError(
        "This feature requires a commercial license of fiedler-optimizer. "
        "See the project README for details."
    )

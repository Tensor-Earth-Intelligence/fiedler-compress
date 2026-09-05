"""
Spectral certificate generation and verification.

A spectral certificate is a cryptographic attestation that a compressed
output was produced by a specific Fiedler spectral partition and has not
been tampered with.  Certificates contain only hashes and HMAC signatures
-- never raw text or proprietary algorithmic constants.

The certificate binds four spectral fingerprints together:
  1. Fiedler vector hash  (sign-canonicalized, rounded)
  2. Laplacian eigenvalue spectrum hash  (full spectrum, sorted)
  3. Zone boundary hash  (zone assignments with confidence)
  4. Compressed output hash  (SHA-256 of UTF-8 bytes)

These are then signed with HMAC-SHA256 using a user-provided or
auto-generated key.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import re
import secrets
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

_ALLOWED_CERT_FIELDS = frozenset({
    "version",
    "timestamp",
    "fiedler_hash",
    "spectrum_hash",
    "zone_hash",
    "compressed_hash",
    "signature",
})

_ALLOWED_PROVENANCE_FIELDS = frozenset({
    "version",
    "timestamp",
    "prompt_commitment",
    "spectrum_hash",
    "zone_hash",
    "fiedler_hash",
    "compressed_hash",
    "signature",
})

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CERTIFICATE_VERSION = "1.0"
PROVENANCE_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralCertificate:
    """
    Strict schema for a spectral compression certificate.

    Every field is either a control string or a hex-encoded hash/signature.
    No raw text is ever stored.

    Parameters
    ----------
    version : str
        Certificate schema version.
    timestamp : str
        ISO 8601 UTC timestamp of certificate creation.
    fiedler_hash : str
        SHA-256 hex digest of the canonicalized Fiedler vector.
    spectrum_hash : str
        SHA-256 hex digest of the sorted Laplacian eigenvalue spectrum.
    zone_hash : str
        SHA-256 hex digest of the serialized zone assignments.
    compressed_hash : str
        SHA-256 hex digest of the compressed output text.
    signature : str
        HMAC-SHA256 hex digest binding all fields above.
    """

    version: str
    timestamp: str
    fiedler_hash: str
    spectrum_hash: str
    zone_hash: str
    compressed_hash: str
    signature: str

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")

        # Validate timestamp is parseable ISO 8601
        try:
            datetime.fromisoformat(self.timestamp)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"timestamp is not valid ISO 8601: {self.timestamp!r}") from exc

        # All hash and signature fields must be exactly 64 lowercase hex chars
        for name in ("fiedler_hash", "spectrum_hash", "zone_hash",
                      "compressed_hash", "signature"):
            value = getattr(self, name)
            if not _HEX64_RE.match(value):
                raise ValueError(
                    f"{name} must be exactly 64 lowercase hex characters, "
                    f"got {len(value)} chars: {value[:20]!r}..."
                )


@dataclass(frozen=True)
class ProvenanceCertificate:
    """
    Chain-of-custody certificate for prompt provenance verification.

    Extends the spectral certificate concept with a *content-blind commitment*:
    a SHA-256 hash of the original prompt.  A holder can later prove "I sent
    this exact prompt at this time" without revealing the prompt to a verifier.

    The verifier only needs the certificate and the original prompt bytes to
    call :func:`verify_provenance`.  The compressed output is **not** required
    for provenance checks — only for full spectral verification.

    Fields
    ------
    version : str
        Provenance schema version.
    timestamp : str
        ISO 8601 UTC timestamp of compression.
    prompt_commitment : str
        SHA-256 hex digest of the original prompt (content-blind commitment).
    spectrum_hash : str
        SHA-256 hex digest of the sorted Laplacian eigenvalue spectrum.
    zone_hash : str
        SHA-256 hex digest of serialized zone assignments.
    fiedler_hash : str
        SHA-256 hex digest of the canonicalized Fiedler vector.
    compressed_hash : str
        SHA-256 hex digest of the compressed output text.
    signature : str
        HMAC-SHA256 hex digest binding all fields above.
    """

    version: str
    timestamp: str
    prompt_commitment: str
    spectrum_hash: str
    zone_hash: str
    fiedler_hash: str
    compressed_hash: str
    signature: str

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")

        try:
            datetime.fromisoformat(self.timestamp)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"timestamp is not valid ISO 8601: {self.timestamp!r}") from exc

        for name in ("prompt_commitment", "spectrum_hash", "zone_hash",
                      "fiedler_hash", "compressed_hash", "signature"):
            value = getattr(self, name)
            if not _HEX64_RE.match(value):
                raise ValueError(
                    f"{name} must be exactly 64 lowercase hex characters, "
                    f"got {len(value)} chars: {value[:20]!r}..."
                )


def validate_provenance_certificate(cert: ProvenanceCertificate) -> bool:
    """
    Validate that *cert* conforms to the provenance allowlisted schema.

    Raises ``ValueError`` if any field name is not in the allowlist.
    Returns ``True`` on success.
    """
    for f in fields(cert):
        if f.name not in _ALLOWED_PROVENANCE_FIELDS:
            raise ValueError(f"Disallowed field in provenance certificate schema: {f.name}")
    ProvenanceCertificate(**asdict(cert))
    return True


def validate_certificate(cert: SpectralCertificate) -> bool:
    """
    Validate that *cert* conforms to the allowlisted schema.

    Raises ``ValueError`` if any field name is not in the allowlist.
    Returns ``True`` on success.
    """
    for f in fields(cert):
        if f.name not in _ALLOWED_CERT_FIELDS:
            raise ValueError(f"Disallowed field in certificate schema: {f.name}")
    # Re-run post-init checks
    SpectralCertificate(**asdict(cert))
    return True


# ---------------------------------------------------------------------------
# Internal hash helpers
# ---------------------------------------------------------------------------

def _canonicalize_fiedler(fiedler: np.ndarray) -> np.ndarray:
    """
    Sign-canonicalize a Fiedler vector.

    Eigenvectors are defined up to a sign flip.  We enforce a convention
    where the first element with magnitude > 1e-12 is positive.  This
    guarantees identical hashes regardless of which eigensolver branch
    was taken.
    """
    for v in fiedler.flat:
        if abs(v) > 1e-12:
            return fiedler * float(np.sign(v))
            break
    return fiedler.copy()


def _hash_array(arr: np.ndarray, precision: int = 10) -> str:
    """SHA-256 hex digest of a deterministically serialized numpy array."""
    rounded = np.round(arr, precision)
    canonical = repr(rounded.tolist()).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _compute_spectrum_hash(adjacency: np.ndarray) -> str:
    """
    Compute SHA-256 of the full sorted Laplacian eigenvalue spectrum.

    Rebuilds the Laplacian L = D - A from the adjacency matrix and uses
    the dense solver ``np.linalg.eigh`` (guaranteed ascending order).
    The matrix is at most 500x500 per MAX_CHUNKS in graph.py.
    """
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigenvalues = np.linalg.eigh(laplacian)[0]  # ascending order
    return _hash_array(eigenvalues)


def _hash_zones(zoned: Sequence) -> str:
    """
    SHA-256 of a deterministic serialization of zone assignments.

    Each ZonedChunk contributes ``(chunk.index, zone.name, confidence)``
    rounded to 6 decimal places.
    """
    tuples = [
        (zc.chunk.index, zc.zone.name, round(zc.confidence, 6))
        for zc in zoned
    ]
    canonical = repr(tuples).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _hash_text(text: str) -> str:
    """SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_hmac_message(
    version: str,
    timestamp: str,
    fiedler_hash: str,
    spectrum_hash: str,
    zone_hash: str,
    compressed_hash: str,
) -> str:
    """Build the pipe-delimited message that gets HMAC-signed."""
    return "|".join([
        version, timestamp,
        fiedler_hash, spectrum_hash, zone_hash, compressed_hash,
    ])


def _compute_hmac(message: str, key_hex: str) -> str:
    """Compute HMAC-SHA256 over *message* using *key_hex* (hex-encoded key)."""
    return _hmac.new(
        bytes.fromhex(key_hex),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Public API — generation
# ---------------------------------------------------------------------------

def generate_certificate(
    fiedler_vector: np.ndarray,
    adjacency: np.ndarray,
    zoned: Sequence,
    compressed: str,
    signing_key: str | None = None,
) -> tuple[SpectralCertificate, str]:
    """
    Generate a spectral certificate for a compressed output.

    Parameters
    ----------
    fiedler_vector : np.ndarray
        The Fiedler vector from spectral decomposition.
    adjacency : np.ndarray
        The (possibly ligature-enhanced) adjacency matrix.
    zoned : Sequence[ZonedChunk]
        Zone-classified chunks.
    compressed : str
        The compressed output text.
    signing_key : str, optional
        Hex-encoded HMAC signing key (64 hex chars = 256 bits).
        If ``None``, a key is auto-generated via ``secrets.token_hex(32)``.

    Returns
    -------
    certificate : SpectralCertificate
        The generated certificate.
    signing_key : str
        The signing key (returned so auto-generated keys are accessible).
    """
    if signing_key is None:
        signing_key = secrets.token_hex(32)

    # Compute hashes — never from raw text, only spectral/structural data
    canon_fiedler = _canonicalize_fiedler(fiedler_vector)
    fiedler_hash = _hash_array(canon_fiedler)
    spectrum_hash = _compute_spectrum_hash(adjacency)
    zone_hash = _hash_zones(zoned)
    compressed_hash = _hash_text(compressed)

    version = CERTIFICATE_VERSION
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build and sign
    message = _build_hmac_message(
        version, timestamp, fiedler_hash, spectrum_hash,
        zone_hash, compressed_hash,
    )
    signature = _compute_hmac(message, signing_key)

    cert = SpectralCertificate(
        version=version,
        timestamp=timestamp,
        fiedler_hash=fiedler_hash,
        spectrum_hash=spectrum_hash,
        zone_hash=zone_hash,
        compressed_hash=compressed_hash,
        signature=signature,
    )

    return cert, signing_key


# ---------------------------------------------------------------------------
# Public API — verification
# ---------------------------------------------------------------------------

def verify_certificate(
    certificate: SpectralCertificate,
    compressed: str,
    signing_key: str,
) -> bool:
    """
    Verify a spectral certificate against compressed output.

    Checks:
    1. The compressed text hash matches the certificate's ``compressed_hash``.
    2. The HMAC signature is valid for the given signing key.

    Uses constant-time comparison via ``hmac.compare_digest``.

    Parameters
    ----------
    certificate : SpectralCertificate
        The certificate to verify.
    compressed : str
        The compressed text to verify against.
    signing_key : str
        The hex-encoded HMAC signing key.

    Returns
    -------
    bool
        ``True`` if the certificate is valid, ``False`` otherwise.
    """
    try:
        validate_certificate(certificate)
    except ValueError:
        return False

    # Check compressed text integrity
    expected_hash = _hash_text(compressed)
    if not _hmac.compare_digest(expected_hash, certificate.compressed_hash):
        return False

    # Rebuild HMAC and verify signature
    message = _build_hmac_message(
        certificate.version,
        certificate.timestamp,
        certificate.fiedler_hash,
        certificate.spectrum_hash,
        certificate.zone_hash,
        certificate.compressed_hash,
    )
    expected_sig = _compute_hmac(message, signing_key)
    return _hmac.compare_digest(expected_sig, certificate.signature)


# ---------------------------------------------------------------------------
# Public API — provenance generation
# ---------------------------------------------------------------------------

def _build_provenance_hmac_message(
    version: str,
    timestamp: str,
    prompt_commitment: str,
    spectrum_hash: str,
    zone_hash: str,
    fiedler_hash: str,
    compressed_hash: str,
) -> str:
    """Build the pipe-delimited message that gets HMAC-signed for provenance."""
    return "|".join([
        version, timestamp, prompt_commitment,
        spectrum_hash, zone_hash, fiedler_hash, compressed_hash,
    ])


def generate_provenance_certificate(
    original_prompt: str,
    fiedler_vector: np.ndarray,
    adjacency: np.ndarray,
    zoned: Sequence,
    compressed: str,
    signing_key: str | None = None,
) -> tuple[ProvenanceCertificate, str]:
    """
    Generate a provenance certificate for chain-of-custody verification.

    The certificate contains a content-blind commitment (SHA-256 of the
    original prompt) and a spectral fingerprint that uniquely identifies
    the compression topology.  No recoverable original text is stored.

    Parameters
    ----------
    original_prompt : str
        The original, uncompressed prompt text.
    fiedler_vector : np.ndarray
        The Fiedler vector from spectral decomposition.
    adjacency : np.ndarray
        The (possibly ligature-enhanced) adjacency matrix.
    zoned : Sequence[ZonedChunk]
        Zone-classified chunks.
    compressed : str
        The compressed output text.
    signing_key : str, optional
        Hex-encoded HMAC signing key (64 hex chars = 256 bits).
        If ``None``, a key is auto-generated.

    Returns
    -------
    certificate : ProvenanceCertificate
        The generated provenance certificate.
    signing_key : str
        The signing key used.
    """
    if signing_key is None:
        signing_key = secrets.token_hex(32)

    canon_fiedler = _canonicalize_fiedler(fiedler_vector)
    fiedler_hash = _hash_array(canon_fiedler)
    spectrum_hash = _compute_spectrum_hash(adjacency)
    zone_hash = _hash_zones(zoned)
    compressed_hash = _hash_text(compressed)
    prompt_commitment = _hash_text(original_prompt)

    version = PROVENANCE_VERSION
    timestamp = datetime.now(timezone.utc).isoformat()

    message = _build_provenance_hmac_message(
        version, timestamp, prompt_commitment,
        spectrum_hash, zone_hash, fiedler_hash, compressed_hash,
    )
    signature = _compute_hmac(message, signing_key)

    cert = ProvenanceCertificate(
        version=version,
        timestamp=timestamp,
        prompt_commitment=prompt_commitment,
        spectrum_hash=spectrum_hash,
        zone_hash=zone_hash,
        fiedler_hash=fiedler_hash,
        compressed_hash=compressed_hash,
        signature=signature,
    )

    return cert, signing_key


# ---------------------------------------------------------------------------
# Public API — provenance verification
# ---------------------------------------------------------------------------

def verify_provenance(
    certificate: ProvenanceCertificate,
    original_prompt: str,
    signing_key: str,
) -> bool:
    """
    Verify that an original prompt matches the provenance commitment.

    This does **not** require the compressed output — only the original
    prompt and the signing key.  The verifier confirms:

    1. The prompt's SHA-256 matches ``prompt_commitment``.
    2. The HMAC signature is valid for the given key.

    Parameters
    ----------
    certificate : ProvenanceCertificate
        The provenance certificate to verify.
    original_prompt : str
        The claimed original prompt.
    signing_key : str
        The hex-encoded HMAC signing key.

    Returns
    -------
    bool
        ``True`` if the prompt matches the commitment and signature is valid.
    """
    try:
        validate_provenance_certificate(certificate)
    except ValueError:
        return False

    expected_commitment = _hash_text(original_prompt)
    if not _hmac.compare_digest(expected_commitment, certificate.prompt_commitment):
        return False

    message = _build_provenance_hmac_message(
        certificate.version,
        certificate.timestamp,
        certificate.prompt_commitment,
        certificate.spectrum_hash,
        certificate.zone_hash,
        certificate.fiedler_hash,
        certificate.compressed_hash,
    )
    expected_sig = _compute_hmac(message, signing_key)
    return _hmac.compare_digest(expected_sig, certificate.signature)


# ---------------------------------------------------------------------------
# Public API — serialization
# ---------------------------------------------------------------------------

def certificate_to_json(cert: SpectralCertificate) -> str:
    """Serialize a SpectralCertificate to a JSON string."""
    return json.dumps(asdict(cert), indent=2)


def certificate_from_json(data: str | dict) -> SpectralCertificate:
    """
    Deserialize a SpectralCertificate from a JSON string or dict.

    Raises ``ValueError`` if there are extra or missing fields.
    """
    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    extra = set(data.keys()) - _ALLOWED_CERT_FIELDS
    if extra:
        raise ValueError(f"Extra fields not allowed in certificate: {extra}")

    missing = _ALLOWED_CERT_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required certificate fields: {missing}")

    return SpectralCertificate(**data)


def provenance_to_json(cert: ProvenanceCertificate) -> str:
    """Serialize a ProvenanceCertificate to a JSON string."""
    return json.dumps(asdict(cert), indent=2)


def provenance_from_json(data: str | dict) -> ProvenanceCertificate:
    """
    Deserialize a ProvenanceCertificate from a JSON string or dict.

    Raises ``ValueError`` if there are extra or missing fields.
    """
    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    extra = set(data.keys()) - _ALLOWED_PROVENANCE_FIELDS
    if extra:
        raise ValueError(f"Extra fields not allowed in provenance certificate: {extra}")

    missing = _ALLOWED_PROVENANCE_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required provenance fields: {missing}")

    return ProvenanceCertificate(**data)

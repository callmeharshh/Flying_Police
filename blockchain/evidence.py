"""Canonical evidence payloads for on-chain anchoring."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional


SEVERITY_TO_CHAIN_VALUE = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


@dataclass(frozen=True)
class EvidencePayload:
    """A deterministic, off-chain evidence record anchored by hash on Monad."""

    frame_id: int
    timestamp: str
    location: str
    rule_id: str
    severity: str
    message: str
    description: str
    objects: tuple[str, ...]
    bbox: Optional[tuple[int, int, int, int]] = None
    track_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "flying-police.alert.v1",
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "location": self.location,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "description": self.description,
            "objects": list(self.objects),
        }
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        if self.track_id:
            payload["track_id"] = self.track_id
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def severity_value(self) -> int:
        return SEVERITY_TO_CHAIN_VALUE.get(self.severity.lower(), 0)

    def evidence_hash(self) -> str:
        return keccak256_hex(self.canonical_json().encode("utf-8"))


def build_alert_evidence(
    *,
    frame_id: int,
    timestamp: str,
    location: str,
    rule_id: str,
    severity: str,
    message: str,
    description: str,
    objects: list[str],
    bbox: Optional[tuple[int, int, int, int]] = None,
    track_id: Optional[str] = None,
) -> EvidencePayload:
    return EvidencePayload(
        frame_id=frame_id,
        timestamp=timestamp,
        location=location,
        rule_id=rule_id,
        severity=severity.lower(),
        message=message,
        description=description,
        objects=tuple(sorted(objects)),
        bbox=tuple(int(v) for v in bbox) if bbox is not None else None,
        track_id=track_id,
    )


def keccak256_hex(data: bytes) -> str:
    """Return an EVM-compatible keccak hash when web3/pycryptodome is installed."""
    try:
        from web3 import Web3

        return Web3.keccak(data).hex()
    except Exception:
        pass

    try:
        from Crypto.Hash import keccak

        digest = keccak.new(digest_bits=256)
        digest.update(data)
        return "0x" + digest.hexdigest()
    except Exception:
        # Offline fallback for local previews/tests. Live contract anchoring requires
        # web3 so the hash matches Solidity keccak256 exactly.
        return "0x" + hashlib.sha3_256(data).hexdigest()


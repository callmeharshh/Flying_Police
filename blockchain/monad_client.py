"""Optional Monad EvidenceRegistry client."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blockchain.evidence import EvidencePayload
from blockchain.lighthouse_client import LighthouseClient
from config import (
    EVIDENCE_REGISTRY_ADDRESS,
    MONAD_CHAIN_ID,
    MONAD_EXPLORER_TX_URL,
    MONAD_PRIVATE_KEY,
    MONAD_RPC_URL,
)


EVIDENCE_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"},
            {"internalType": "uint64", "name": "frameId", "type": "uint64"},
            {"internalType": "uint8", "name": "severity", "type": "uint8"},
            {"internalType": "string", "name": "location", "type": "string"},
            {"internalType": "string", "name": "message", "type": "string"},
            {"internalType": "string", "name": "ipfsCid", "type": "string"},
        ],
        "name": "anchor",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"}],
        "name": "verify",
        "outputs": [
            {"internalType": "bool", "name": "exists", "type": "bool"},
            {"internalType": "uint64", "name": "anchoredAt", "type": "uint64"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


@dataclass(frozen=True)
class AnchorResult:
    status: str
    evidence_hash: str
    tx_hash: Optional[str] = None
    explorer_url: Optional[str] = None
    message: str = ""
    location: str = ""
    alert_message: str = ""
    frame_id: int = 0
    ipfs_cid: Optional[str] = None
    ipfs_gateway_url: Optional[str] = None

    @property
    def anchored(self) -> bool:
        return self.status == "anchored"


@dataclass(frozen=True)
class VerifyResult:
    status: str
    evidence_hash: str
    exists: bool = False
    anchored_at: int = 0
    message: str = ""


class MonadEvidenceClient:
    """Upload evidence to Lighthouse IPFS, then anchor metadata on Monad."""

    def __init__(
        self,
        *,
        rpc_url: str = MONAD_RPC_URL,
        chain_id: int = MONAD_CHAIN_ID,
        private_key: str = MONAD_PRIVATE_KEY,
        contract_address: str = EVIDENCE_REGISTRY_ADDRESS,
        explorer_tx_url: str = MONAD_EXPLORER_TX_URL,
        lighthouse_client: Optional[LighthouseClient] = None,
    ):
        self.rpc_url = rpc_url
        self.chain_id = chain_id
        self.private_key = private_key
        self.contract_address = contract_address
        self.explorer_tx_url = explorer_tx_url
        self._lighthouse = lighthouse_client or LighthouseClient()
        self._web3 = None
        self._account = None
        self._contract = None

    def is_configured(self) -> bool:
        return bool(
            self.rpc_url
            and self.chain_id
            and self.private_key
            and self.contract_address
        )

    def anchor(self, evidence: EvidencePayload) -> AnchorResult:
        evidence_hash = evidence.evidence_hash()
        base_kwargs = {
            "evidence_hash": evidence_hash,
            "location": evidence.location,
            "alert_message": evidence.message,
            "frame_id": evidence.frame_id,
        }
        if not self.is_configured():
            return AnchorResult(
                status="not_configured",
                message=(
                    "Monad anchoring skipped — set MONAD_RPC_URL, MONAD_CHAIN_ID, "
                    "MONAD_PRIVATE_KEY, and EVIDENCE_REGISTRY_ADDRESS."
                ),
                **base_kwargs,
            )

        upload_result = self._lighthouse.upload_json(evidence.canonical_json())
        ipfs_cid = upload_result.cid or ""
        if not upload_result.uploaded:
            return AnchorResult(
                status="ipfs_error",
                message=upload_result.message,
                ipfs_cid=upload_result.cid,
                ipfs_gateway_url=upload_result.gateway_url,
                **base_kwargs,
            )

        try:
            self._ensure_contract()
            tx = self._contract.functions.anchor(
                bytes.fromhex(evidence_hash.removeprefix("0x")),
                int(evidence.frame_id),
                evidence.severity_value(),
                evidence.location,
                evidence.message,
                ipfs_cid,
            ).build_transaction({
                "from": self._account.address,
                "nonce": self._web3.eth.get_transaction_count(self._account.address),
                "chainId": self.chain_id,
            })
            signed = self._account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed.raw_transaction).hex()
            explorer_url = self._explorer_url(tx_hash)
            return AnchorResult(
                status="anchored",
                tx_hash=tx_hash,
                explorer_url=explorer_url,
                message="Evidence uploaded to Lighthouse IPFS and anchored on Monad.",
                ipfs_cid=ipfs_cid,
                ipfs_gateway_url=upload_result.gateway_url,
                **base_kwargs,
            )
        except Exception as exc:
            return AnchorResult(
                status="error",
                message=f"Monad anchoring failed: {exc}",
                ipfs_cid=ipfs_cid,
                ipfs_gateway_url=upload_result.gateway_url,
                **base_kwargs,
            )

    def verify(self, evidence: EvidencePayload) -> VerifyResult:
        evidence_hash = evidence.evidence_hash()
        if not self.is_configured():
            return VerifyResult(
                status="not_configured",
                evidence_hash=evidence_hash,
                message="Monad verification skipped — client is not configured.",
            )

        try:
            self._ensure_contract()
            exists, anchored_at = self._contract.functions.verify(
                bytes.fromhex(evidence_hash.removeprefix("0x"))
            ).call()
            return VerifyResult(
                status="verified",
                evidence_hash=evidence_hash,
                exists=bool(exists),
                anchored_at=int(anchored_at),
            )
        except Exception as exc:
            return VerifyResult(
                status="error",
                evidence_hash=evidence_hash,
                message=f"Monad verification failed: {exc}",
            )

    def _ensure_contract(self) -> None:
        if self._contract is not None:
            return

        try:
            from web3 import Web3
        except ImportError as exc:
            raise RuntimeError(
                "web3 is required for live Monad anchoring. Install requirements.txt."
            ) from exc

        self._web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self._web3.is_connected():
            raise RuntimeError("could not connect to Monad RPC")

        self._account = self._web3.eth.account.from_key(self.private_key)
        checksum_address = self._web3.to_checksum_address(self.contract_address)
        self._contract = self._web3.eth.contract(
            address=checksum_address,
            abi=EVIDENCE_REGISTRY_ABI,
        )

    def _explorer_url(self, tx_hash: str) -> Optional[str]:
        if not self.explorer_tx_url:
            return None
        return f"{self.explorer_tx_url.rstrip('/')}/{tx_hash}"

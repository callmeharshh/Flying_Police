from unittest.mock import MagicMock, patch

from blockchain.evidence import build_alert_evidence
from blockchain.lighthouse_client import LighthouseClient
from blockchain.monad_client import MonadEvidenceClient
from storage.event_store import EventStore


def test_evidence_payload_is_canonical_for_same_alert():
    first = build_alert_evidence(
        frame_id=7,
        timestamp="2026-07-04T10:30:00+05:30",
        location="perimeter",
        rule_id="RULE-05",
        severity="HIGH",
        message="Activity detected at perimeter at night",
        description="a person near the fence",
        objects=["vehicle", "person"],
        bbox=(10, 20, 30, 40),
        track_id="track_001",
    )
    second = build_alert_evidence(
        frame_id=7,
        timestamp="2026-07-04T10:30:00+05:30",
        location="perimeter",
        rule_id="RULE-05",
        severity="high",
        message="Activity detected at perimeter at night",
        description="a person near the fence",
        objects=["person", "vehicle"],
        bbox=(10, 20, 30, 40),
        track_id="track_001",
    )

    assert first.canonical_json() == second.canonical_json()
    assert first.evidence_hash() == second.evidence_hash()
    assert first.severity_value() == 2


def test_monad_client_skips_when_not_configured():
    evidence = build_alert_evidence(
        frame_id=1,
        timestamp="2026-07-04T10:30:00+05:30",
        location="main_gate",
        rule_id="RULE-01",
        severity="high",
        message="Person detected at night at main_gate",
        description="a person at the gate",
        objects=["person"],
    )
    client = MonadEvidenceClient(
        rpc_url="",
        chain_id=0,
        private_key="",
        contract_address="",
    )

    result = client.anchor(evidence)

    assert result.status == "not_configured"
    assert result.tx_hash is None
    assert len(result.evidence_hash.replace("0x", "")) == 64
    assert result.location == "main_gate"
    assert result.alert_message == "Person detected at night at main_gate"


def test_monad_client_returns_ipfs_error_when_lighthouse_not_configured():
    evidence = build_alert_evidence(
        frame_id=2,
        timestamp="2026-07-04T10:30:00+05:30",
        location="garage",
        rule_id="RULE-02",
        severity="medium",
        message="Unknown vehicle detected",
        description="a car in the garage",
        objects=["vehicle"],
    )
    client = MonadEvidenceClient(
        rpc_url="https://testnet-rpc.monad.xyz",
        chain_id=10143,
        private_key="0x" + "11" * 32,
        contract_address="0x" + "22" * 20,
        lighthouse_client=LighthouseClient(api_key=""),
    )

    result = client.anchor(evidence)

    assert result.status == "ipfs_error"
    assert result.tx_hash is None
    assert result.location == "garage"


@patch("blockchain.lighthouse_client.requests.post")
def test_lighthouse_client_uploads_json(mock_post):
    mock_response = MagicMock()
    mock_response.text = '{"Name":"evidence.json","Hash":"QmTestHash123","Size":"120"}\n'
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    client = LighthouseClient(api_key="test-key")
    result = client.upload_json('{"schema":"airsecure.alert.v1"}')

    assert result.uploaded
    assert result.cid == "QmTestHash123"
    assert result.gateway_url.endswith("/QmTestHash123")
    mock_post.assert_called_once()


def test_event_store_records_evidence_anchor(tmp_path):
    store = EventStore(db_path=str(tmp_path / "events.db"))

    anchor_id = store.log_evidence_anchor(
        frame_id=3,
        evidence_hash="0xabc",
        tx_hash="0xdef",
        status="anchored",
        message="Evidence anchored on Monad.",
        location="main_gate",
        alert_message="Person loitering at main_gate",
        ipfs_cid="QmTestHash123",
    )
    anchors = store.get_evidence_anchors()

    assert anchor_id.startswith("anch_")
    assert len(anchors) == 1
    assert anchors[0]["evidence_hash"] == "0xabc"
    assert anchors[0]["tx_hash"] == "0xdef"
    assert anchors[0]["status"] == "anchored"
    assert anchors[0]["location"] == "main_gate"
    assert anchors[0]["alert_message"] == "Person loitering at main_gate"
    assert anchors[0]["ipfs_cid"] == "QmTestHash123"

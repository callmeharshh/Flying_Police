import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional
from config import EVENTS_DB_PATH


class EventStore:
    def __init__(self, db_path: str = None):
        self._conn = sqlite3.connect(db_path or EVENTS_DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id   TEXT PRIMARY KEY,
                frame_id   INTEGER,
                timestamp  TEXT,
                type       TEXT,
                message    TEXT,
                severity   TEXT
            );
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id   TEXT PRIMARY KEY,
                event_id   TEXT,
                frame_id   INTEGER,
                timestamp  TEXT,
                rule_id    TEXT,
                message    TEXT,
                severity   TEXT,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            );
            CREATE TABLE IF NOT EXISTS evidence_anchors (
                anchor_id     TEXT PRIMARY KEY,
                event_id      TEXT,
                frame_id      INTEGER,
                timestamp     TEXT,
                evidence_hash TEXT,
                tx_hash       TEXT,
                status        TEXT,
                message       TEXT,
                location      TEXT,
                alert_message TEXT,
                ipfs_cid      TEXT,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            );
        """)
        self._conn.commit()
        self._migrate_evidence_anchor_columns()

    def _migrate_evidence_anchor_columns(self) -> None:
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(evidence_anchors)").fetchall()
        }
        migrations = {
            "location": "ALTER TABLE evidence_anchors ADD COLUMN location TEXT DEFAULT ''",
            "alert_message": "ALTER TABLE evidence_anchors ADD COLUMN alert_message TEXT DEFAULT ''",
            "ipfs_cid": "ALTER TABLE evidence_anchors ADD COLUMN ipfs_cid TEXT DEFAULT ''",
        }
        for column, statement in migrations.items():
            if column not in existing:
                self._conn.execute(statement)
        self._conn.commit()

    def log_event(self, frame_id: int, message: str, severity: str = "low", type: str = "log") -> str:
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, frame_id, timestamp, type, message, severity)
        )
        self._conn.commit()
        return event_id

    def log_alert(self, frame_id: int, rule_id: str, message: str, severity: str) -> str:
        event_id = self.log_event(frame_id, message, severity, type="alert")
        alert_id = f"alrt_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (alert_id, event_id, frame_id, timestamp, rule_id, message, severity)
        )
        self._conn.commit()
        return alert_id

    def log_evidence_anchor(
        self,
        frame_id: int,
        evidence_hash: str,
        status: str,
        message: str = "",
        tx_hash: Optional[str] = None,
        event_id: Optional[str] = None,
        location: str = "",
        alert_message: str = "",
        ipfs_cid: Optional[str] = None,
    ) -> str:
        anchor_id = f"anch_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO evidence_anchors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                anchor_id,
                event_id,
                frame_id,
                timestamp,
                evidence_hash,
                tx_hash,
                status,
                message,
                location,
                alert_message,
                ipfs_cid or "",
            ),
        )
        self._conn.commit()
        return anchor_id

    def get_alerts(self, severity: Optional[str] = None) -> List[dict]:
        if severity:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE severity = ? ORDER BY timestamp", (severity,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM alerts ORDER BY timestamp").fetchall()
        return [dict(r) for r in rows]

    def get_events_by_timerange(self, start: str, end: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (start, end)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_events(self) -> List[dict]:
        rows = self._conn.execute("SELECT * FROM events ORDER BY timestamp").fetchall()
        return [dict(r) for r in rows]

    def get_evidence_anchors(self) -> List[dict]:
        rows = self._conn.execute("SELECT * FROM evidence_anchors ORDER BY timestamp").fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self._conn.close()

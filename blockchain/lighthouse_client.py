"""Upload evidence JSON to Lighthouse IPFS storage."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Optional

import requests

from config import (
    LIGHTHOUSE_API_KEY,
    LIGHTHOUSE_GATEWAY_URL,
    LIGHTHOUSE_UPLOAD_TIMEOUT_SECONDS,
    LIGHTHOUSE_UPLOAD_URL,
)


@dataclass(frozen=True)
class LighthouseUploadResult:
    status: str
    cid: Optional[str] = None
    gateway_url: Optional[str] = None
    message: str = ""

    @property
    def uploaded(self) -> bool:
        return self.status == "uploaded"


class LighthouseClient:
    """Thin wrapper around the Lighthouse upload API."""

    def __init__(
        self,
        *,
        api_key: str = LIGHTHOUSE_API_KEY,
        upload_url: str = LIGHTHOUSE_UPLOAD_URL,
        gateway_url: str = LIGHTHOUSE_GATEWAY_URL,
        timeout_seconds: int = LIGHTHOUSE_UPLOAD_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key
        self.upload_url = upload_url
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def upload_json(self, content: str, *, filename: str = "evidence.json") -> LighthouseUploadResult:
        if not self.is_configured():
            return LighthouseUploadResult(
                status="not_configured",
                message=(
                    "Lighthouse upload skipped — set LIGHTHOUSE_API_KEY "
                    "(create one at https://files.lighthouse.storage/)."
                ),
            )

        try:
            response = requests.post(
                self.upload_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Encryption": "false",
                },
                files={
                    "file": (filename, io.BytesIO(content.encode("utf-8")), "application/json"),
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            last_line = response.text.strip().split("\n")[-1]
            payload = json.loads(last_line)
            cid = payload.get("Hash")
            if not cid:
                return LighthouseUploadResult(
                    status="error",
                    message="Lighthouse upload succeeded but no CID was returned.",
                )
            return LighthouseUploadResult(
                status="uploaded",
                cid=cid,
                gateway_url=f"{self.gateway_url}/{cid}",
                message="Evidence JSON uploaded to Lighthouse IPFS.",
            )
        except Exception as exc:
            return LighthouseUploadResult(
                status="error",
                message=f"Lighthouse upload failed: {exc}",
            )

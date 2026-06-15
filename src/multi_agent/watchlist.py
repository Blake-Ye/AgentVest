from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WatchlistStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def upsert(self, entry: dict[str, Any]) -> dict[str, Any]:
        payload = self._read_payload()
        items = payload["items"]
        normalized_entry = dict(entry)
        normalized_entry["updated_at"] = datetime.now(timezone.utc).isoformat()

        for index, existing in enumerate(items):
            if existing.get("company_ticker") == normalized_entry.get("company_ticker"):
                items[index] = normalized_entry
                self._write_payload(payload)
                return normalized_entry

        items.append(normalized_entry)
        self._write_payload(payload)
        return normalized_entry

    def list_items(self) -> list[dict[str, Any]]:
        payload = self._read_payload()
        return sorted(
            payload["items"],
            key=lambda item: (-float(item.get("trust_score", 0.0) or 0.0), str(item.get("company_ticker", ""))),
        )

    def _read_payload(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {"items": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write_payload(self, payload: dict[str, list[dict[str, Any]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

import json
from pathlib import Path

from multi_agent.watchlist import WatchlistStore


def test_watchlist_store_upserts_entry_by_ticker(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "watchlist.json")
    first_entry = {
        "company_name": "Alibaba Group Holding Ltd",
        "company_ticker": "BABA",
        "stance": "watch",
        "stance_label": "观察",
        "trust_score": 72.0,
    }
    updated_entry = {
        "company_name": "Alibaba Group Holding Ltd",
        "company_ticker": "BABA",
        "stance": "buy",
        "stance_label": "增持",
        "trust_score": 87.0,
    }

    store.upsert(first_entry)
    store.upsert(updated_entry)

    written = json.loads((tmp_path / "watchlist.json").read_text(encoding="utf-8"))
    assert len(written["items"]) == 1
    assert written["items"][0]["stance"] == "buy"
    assert written["items"][0]["trust_score"] == 87.0


def test_watchlist_store_lists_entries_sorted_by_trust_score_desc(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "watchlist.json")
    store.upsert(
        {
            "company_name": "Alibaba Group Holding Ltd",
            "company_ticker": "BABA",
            "stance": "buy",
            "stance_label": "增持",
            "trust_score": 87.0,
        }
    )
    store.upsert(
        {
            "company_name": "Apple Inc.",
            "company_ticker": "AAPL",
            "stance": "watch",
            "stance_label": "观察",
            "trust_score": 65.0,
        }
    )

    items = store.list_items()

    assert [item["company_ticker"] for item in items] == ["BABA", "AAPL"]

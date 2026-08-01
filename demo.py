#!/usr/bin/env python3
"""Directly test the Nasdaq quote endpoint used by AgentVest."""

from __future__ import annotations

import json
import socket
import sys

import requests


HOST = "api.nasdaq.com"
URL = f"https://{HOST}/api/quote/{{ticker}}/info?assetclass=stocks"


def main() -> int:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").strip().upper()
    url = URL.format(ticker=ticker)
    session = requests.Session()
    headers = {
        "User-Agent": "multi-agent-investment-research vpn-demo",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
    }

    print(f"Ticker: {ticker}")
    print(f"URL: {url}")
    proxy_schemes = sorted(requests.utils.get_environ_proxies(url))
    print(f"Environment proxies: {proxy_schemes or 'none'}")

    try:
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(HOST, 443)})
        print(f"DNS: {', '.join(addresses)}")
        response = session.get(url, headers=headers, timeout=15)
    except (OSError, requests.RequestException) as error:
        print(f"NETWORK_ERROR: {type(error).__name__}: {error}")
        return 1

    print(f"HTTP: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type', 'missing')}")
    if response.status_code != 200:
        print(f"Response preview: {response.text[:300]!r}")
        return 2

    try:
        payload = response.json()
    except ValueError:
        print(f"INVALID_JSON: {response.text[:300]!r}")
        return 3

    primary_data = (payload.get("data") or {}).get("primaryData") or {}
    print(f"Nasdaq status: {json.dumps(payload.get('status'), ensure_ascii=False)}")
    print(f"Price: {primary_data.get('lastSalePrice') or 'missing'}")
    print(f"Timestamp: {primary_data.get('lastTradeTimestamp') or 'missing'}")
    if not primary_data.get("lastSalePrice"):
        print(f"Top-level keys: {sorted(payload)}")
        return 4

    print("RESULT: Nasdaq quote is usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""ssh_key_audit.py

A GitHub Action script that audits SSH key rotation across a fleet of servers.
It connects to each server via a lightweight HTTP endpoint (exposed by a tiny
agent on the target host) that returns the list of authorized public keys and
their last rotation timestamps. The script then checks each key against a
configurable maximum age and produces a JSON report suitable for CI output.

Only the Python standard library and the `requests` package are used.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import requests

# ---------------------------------------------------------------------------
# Configuration – can be overridden with environment variables in the Action
# ---------------------------------------------------------------------------
# Comma‑separated list of server hostnames or IPs to audit
SERVERS = [
    server.strip()
    for server in os.getenv("SSH_AUDIT_SERVERS", "").split(",")
    if server.strip()
]

# Base URL path where the tiny SSH‑key‑exposer agent is listening
AGENT_ENDPOINT = os.getenv("SSH_AUDIT_ENDPOINT", "http://{host}:8000/keys")

# Maximum allowed age for a key before it is considered stale (days)
MAX_KEY_AGE_DAYS = int(os.getenv("SSH_AUDIT_MAX_AGE", "90"))

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = int(os.getenv("SSH_AUDIT_TIMEOUT", "5"))

# Path where the final report will be written (GitHub Actions output)
REPORT_PATH = os.getenv("SSH_AUDIT_REPORT", "ssh_key_audit_report.json")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def fetch_keys(host: str) -> List[Dict[str, Any]]:
    """Retrieve the list of SSH keys from a single host.

    The remote agent is expected to return JSON in the form:
    [{"key": "ssh-rsa AAA...", "last_rotated": "2023-07-01T12:34:56Z"}, ...]

    Args:
        host: Hostname or IP address of the target server.

    Returns:
        A list of dictionaries with ``key`` and ``last_rotated`` fields.
    """
    url = AGENT_ENDPOINT.format(host=host)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        logging.info("Fetched %d keys from %s", len(data), host)
        return data
    except requests.RequestException as exc:
        logging.error("Failed to fetch keys from %s: %s", host, exc)
        return []


def is_key_stale(last_rotated: str) -> bool:
    """Determine whether a key is older than the allowed maximum age.

    The ``last_rotated`` string is expected to be ISO‑8601 UTC.

    Args:
        last_rotated: Timestamp string in ISO‑8601 UTC format.

    Returns:
        True if the key is stale, False otherwise.
    """
    try:
        rotated_dt = datetime.strptime(last_rotated, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        logging.warning("Invalid timestamp format: %s", last_rotated)
        return True  # Treat malformed dates as stale
    age = datetime.utcnow() - rotated_dt
    return age.days > MAX_KEY_AGE_DAYS


def main() -> None:
    """Collect stale keys from all configured servers and write a JSON report."""
    report = {}
    for host in SERVERS:
        keys = fetch_keys(host)
        stale_keys = [k for k in keys if is_key_stale(k.get("last_rotated", ""))]
        if stale_keys:
            report[host] = stale_keys

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info("Report written to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
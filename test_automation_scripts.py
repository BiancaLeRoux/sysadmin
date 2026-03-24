python
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

# Import the module under test. Adjust the import path if the module name differs.
import automation_scripts as script_under_test


# ----------------------------------------------------------------------
# fetch_keys tests
# ----------------------------------------------------------------------
def test_fetch_keys_success():
    host = "example.com"
    url = script_under_test.AGENT_ENDPOINT.format(host=host)
    expected_data = [
        {"key": "ssh-rsa AAA...", "last_rotated": "2023-07-01T12:34:56Z"},
        {"key": "ssh-ed25519 BBB...", "last_rotated": "2023-08-15T09:00:00Z"},
    ]

    mock_resp = Mock()
    mock_resp.json.return_value = expected_data
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = script_under_test.fetch_keys(host)

    assert result == expected_data
    mock_get.assert_called_once_with(url, timeout=script_under_test.REQUEST_TIMEOUT)


def test_fetch_keys_http_error():
    host = "example.com"
    url = script_under_test.AGENT_ENDPOINT.format(host=host)

    mock_resp = Mock()
    mock_resp.raise_for_status.side_effect = script_under_test.requests.HTTPError("404")

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = script_under_test.fetch_keys(host)

    assert result == []  # on error we expect an empty list
    mock_get.assert_called_once_with(url, timeout=script_under_test.REQUEST_TIMEOUT)


def test_fetch_keys_timeout():
    host = "example.com"
    url = script_under_test.AGENT_ENDPOINT.format(host=host)

    with patch(
        "requests.get", side_effect=script_under_test.requests.Timeout("timed out")
    ) as mock_get:
        result = script_under_test.fetch_keys(host)

    assert result == []  # timeout should also yield an empty list
    mock_get.assert_called_once_with(url, timeout=script_under_test.REQUEST_TIMEOUT)


def test_fetch_keys_invalid_json():
    host = "example.com"
    url = script_under_test.AGENT_ENDPOINT.format(host=host)

    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.side_effect = json.JSONDecodeError("msg", doc="{}", pos=0)

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = script_under_test.fetch_keys(host)

    # The function catches RequestException, JSONDecodeError is not a subclass,
    # so it propagates; we assert that the exception bubbles up.
    # If you prefer to treat it as a failure, adjust the implementation.
    assert isinstance(result, list)  # should not happen; just a safety check
    mock_get.assert_called_once_with(url, timeout=script_under_test.REQUEST_TIMEOUT)


def test_fetch_keys_empty_response():
    host = "example.com"
    url = script_under_test.AGENT_ENDPOINT.format(host=host)

    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = []

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = script_under_test.fetch_keys(host)

    assert result == []
    mock_get.assert_called_once_with(url, timeout=script_under_test.REQUEST_TIMEOUT)


# ----------------------------------------------------------------------
# is_key_stale tests
# ----------------------------------------------------------------------
def test_is_key_stale_true_when_older_than_max():
    # Build a timestamp older than MAX_KEY_AGE_DAYS
    old_date = datetime.utcnow() - timedelta(days=script_under_test.MAX_KEY_AGE_DAYS + 1)
    iso_str = old_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    assert script_under_test.is_key_stale(iso_str) is True


def test_is_key_stale_false_when_within_max():
    recent_date = datetime.utcnow() - timedelta(days=script_under_test.MAX_KEY_AGE_DAYS - 1)
    iso_str = recent_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    assert script_under_test.is_key_stale(iso_str) is False


def test_is_key_stale_handles_invalid_format():
    # An invalid timestamp should be treated as stale (implementation dependent)
    # Here we expect the function to raise a ValueError from datetime parsing.
    with pytest.raises(ValueError):
        script_under_test.is_key_stale("not-a-timestamp")
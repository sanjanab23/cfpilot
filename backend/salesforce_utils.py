import os
import json
import logging
import time
import threading
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SF_TOKEN_URL    = os.getenv("SF_TOKEN_URL")
SF_CLIENT_ID    = os.getenv("SF_CLIENT_ID")
SF_CLIENT_SECRET = os.getenv("SF_CLIENT_SECRET")
SF_BASE_URL     = os.getenv("SF_BASE_URL")

# ── Timeouts ──────────────────────────────────────────────────────────────────
# Auth is a simple POST — 10s is fine.
# Queries can be slow on large datasets — give them 30s.
AUTH_TIMEOUT  = 10
QUERY_TIMEOUT = 30

# ── Token cache — avoids a new auth call for every query ─────────────────────
# Salesforce tokens are valid for 1 hour (3600s). We refresh at 55 minutes.
_TOKEN_CACHE: dict = {
    "token":      None,
    "expires_at": 0.0,
}
_TOKEN_LOCK = threading.Lock()   # safe for multi-threaded Flask/FastAPI


# ── Pagination cap ────────────────────────────────────────────────────────────
# Max pages to follow for nextRecordsUrl. Each page = up to 2000 records.
# 5 pages = up to 10,000 records — enough for any dashboard query.
MAX_PAGES = 5


def _strip_attributes(obj):
    """
    Recursively remove Salesforce 'attributes' metadata keys from
    records and all nested relationship objects.

    Before:  {"attributes": {"type": "Account"}, "Name": "Tata", "Owner": {"attributes": {...}, "Name": "John"}}
    After:   {"Name": "Tata", "Owner": {"Name": "John"}}
    """
    if isinstance(obj, dict):
        return {
            k: _strip_attributes(v)
            for k, v in obj.items()
            if k != "attributes"
        }
    if isinstance(obj, list):
        return [_strip_attributes(item) for item in obj]
    return obj


def get_salesforce_token() -> Optional[str]:
    """
    Return a cached Salesforce access token, refreshing only when
    the cached token is expired (or missing).

    Tokens are valid for 3600s; we treat them as expired after 3300s
    (55 minutes) to give a 5-minute safety margin.
    """
    with _TOKEN_LOCK:
        now = time.time()

        # Return cached token if still valid
        if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["expires_at"]:
            logger.debug("Using cached Salesforce token")
            return _TOKEN_CACHE["token"]

        # Token missing or expired — fetch a new one
        if not all([SF_TOKEN_URL, SF_CLIENT_ID, SF_CLIENT_SECRET]):
            logger.warning("Salesforce credentials not fully configured")
            return None

        payload = {
            "grant_type":    "client_credentials",
            "client_id":     SF_CLIENT_ID,
            "client_secret": SF_CLIENT_SECRET,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            response = requests.post(
                SF_TOKEN_URL,
                headers=headers,
                data=payload,
                timeout=AUTH_TIMEOUT,
            )
            response.raise_for_status()
            token = response.json().get("access_token")

            if token:
                _TOKEN_CACHE["token"]      = token
                _TOKEN_CACHE["expires_at"] = now + 3300  # 55-minute TTL
                logger.info("Salesforce token refreshed successfully")
                return token
            else:
                logger.error("Salesforce auth response missing access_token")
                return None

        except requests.Timeout:
            logger.error("Salesforce auth request timed out")
            return None
        except Exception as e:
            logger.error(f"Salesforce Auth Error: {e}")
            return None


def execute_soql_query(query: str) -> str:
    """
    Execute a SOQL query against Salesforce and return a valid JSON string.

    Returns:
        - A JSON array string (e.g. '[{"Name": "Tata"}]') on success
        - "[]" when the query returns zero records
        - "Salesforce Error: <message>" on API errors
        - "Error: <message>" on network/config errors

    Improvements over original:
        1. Token caching   — one auth call per 55 minutes, not per query
        2. Pagination      — follows nextRecordsUrl up to MAX_PAGES
        3. json.dumps()    — returns valid JSON, not Python str()
        4. Recursive strip — removes nested 'attributes' keys
        5. "[]" for empty  — not a human-readable string
        6. 30s query timeout — handles slow aggregate queries
    """
    token = get_salesforce_token()
    if not token:
        return "Error: Could not authenticate with Salesforce."

    if not SF_BASE_URL:
        return "Error: Salesforce base URL not configured."

    endpoint = f"{SF_BASE_URL}/query"
    headers  = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }

    all_records = []
    current_url = endpoint
    params      = {"q": query}
    page        = 0

    try:
        logger.info(f"Executing SOQL query (page {page + 1}): {query[:200]}...")

        while current_url and page < MAX_PAGES:
            # First page uses params; subsequent pages use the nextRecordsUrl directly
            if page == 0:
                response = requests.get(
                    current_url,
                    headers=headers,
                    params=params,
                    timeout=QUERY_TIMEOUT,
                )
            else:
                # nextRecordsUrl is a path like /services/data/v59.0/query/01g...
                # Prepend the base domain if it's a relative path
                if current_url.startswith("/"):
                    # Extract just the domain from SF_BASE_URL
                    # SF_BASE_URL is like https://xxx.salesforce.com/services/data/v59.0
                    domain = "/".join(SF_BASE_URL.split("/")[:3])  # https://xxx.salesforce.com
                    current_url = domain + current_url

                response = requests.get(
                    current_url,
                    headers=headers,
                    timeout=QUERY_TIMEOUT,
                )

            # ── Handle non-200 responses ──────────────────────────────────
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    if isinstance(error_data, list) and error_data:
                        msg = error_data[0].get("message", "Unknown Error")
                        error_code = error_data[0].get("errorCode", "")
                        logger.error(f"Salesforce API Error [{error_code}]: {msg}")
                        return f"Salesforce Error: {msg}"
                    return f"Salesforce Error {response.status_code}: {response.text[:200]}"
                except Exception:
                    return f"Salesforce Error {response.status_code}: {response.text[:200]}"

            # ── Parse response ────────────────────────────────────────────
            data    = response.json()
            records = data.get("records", [])

            # Recursively strip all 'attributes' keys (including nested ones)
            cleaned = [_strip_attributes(r) for r in records]
            all_records.extend(cleaned)

            logger.info(
                f"Page {page + 1}: fetched {len(records)} records "
                f"(total so far: {len(all_records)}, "
                f"done={data.get('done', True)})"
            )

            # ── Pagination ────────────────────────────────────────────────
            if data.get("done", True):
                break  # No more pages

            next_url = data.get("nextRecordsUrl")
            if not next_url:
                break

            current_url = next_url
            params      = {}   # params are encoded in nextRecordsUrl already
            page       += 1

        # ── Final result ──────────────────────────────────────────────────
        if not all_records:
            logger.info("Query returned 0 records")
            return "[]"   # ← Always return valid JSON, never a human string

        logger.info(f"Query complete: {len(all_records)} total records returned")

        # json.dumps() produces valid JSON (double quotes, true/false/null)
        return json.dumps(all_records, default=str)

    except requests.Timeout:
        logger.error(f"SOQL query timed out after {QUERY_TIMEOUT}s: {query[:100]}")
        return "Error: Request timed out. Please try again."
    except Exception as e:
        logger.error(f"SOQL Execution Error: {e}")
        return f"Execution Error: {str(e)}"


def invalidate_token_cache() -> None:
    """
    Force the next query to fetch a fresh token.
    Call this if you receive a 401 Unauthorized from Salesforce
    mid-session (e.g. token revoked externally).
    """
    with _TOKEN_LOCK:
        _TOKEN_CACHE["token"]      = None
        _TOKEN_CACHE["expires_at"] = 0.0
        logger.info("Salesforce token cache invalidated")
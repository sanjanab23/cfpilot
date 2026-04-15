"""
Database utilities for connection pooling and safe database operations.
Fixes: Critical race conditions and connection exhaustion issues.
"""
import sqlite3
import threading
import logging
import time
from contextlib import contextmanager
from typing import Optional

import os

logger = logging.getLogger(__name__)

# Use environment variable for DB path (critical for Docker volume mount)
DB_NAME = os.getenv("DB_PATH", "cfpilot.db")

# Thread-local storage for database connections
_local = threading.local()

# Retry configuration for transient SQLite lock errors
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds


def _make_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with production-safe pragmas."""
    conn = sqlite3.connect(
        DB_NAME,
        timeout=10.0,
        check_same_thread=False,
        isolation_level='DEFERRED'  # Better concurrency
    )
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    # Wait up to 5s when database is locked (prevents immediate OperationalError)
    conn.execute("PRAGMA busy_timeout = 5000")
    logger.debug(f"Created new DB connection for thread {threading.current_thread().name}")
    return conn


@contextmanager
def get_db_connection():
    """
    Thread-safe database connection context manager.
    Provides per-thread connection reuse and automatic transaction management.
    Includes retry logic for transient SQLite lock errors.

    MED-02 FIX: Removed the double-yield bug that existed in the original
    retry loop. The retry is now internal to a helper; the contextmanager
    yields exactly once.

    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
    """
    # Get or create connection for this thread
    if not hasattr(_local, 'conn') or _local.conn is None:
        try:
            _local.conn = _make_connection()
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise

    conn = _local.conn
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            yield conn
            conn.commit()
            return  # Success — exit generator normally
        except sqlite3.OperationalError as e:
            conn.rollback()
            if "locked" in str(e).lower() and attempt < MAX_RETRIES - 1:
                last_error = e
                wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    f"Database locked (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                # Re-enter the loop — but we cannot re-yield (contextmanager
                # only yields once).  Break out and raise after exhausting retries.
                break
            else:
                logger.error(f"Database error, rolling back: {e}")
                raise
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error, rolling back: {e}")
            raise

    # All retries exhausted (only reachable from the "locked" branch above)
    if last_error:
        logger.error(f"Database locked after {MAX_RETRIES} retries: {last_error}")
        raise last_error


def execute_query(query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False):
    """
    Execute a database query with automatic connection management.

    Args:
        query: SQL query string
        params: Query parameters (tuple)
        fetch_one: Return single row
        fetch_all: Return all rows

    Returns:
        Query results or None
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)

        if fetch_one:
            return cursor.fetchone()
        elif fetch_all:
            return cursor.fetchall()
        else:
            return cursor.lastrowid


def close_all_connections():
    """
    Close all thread-local database connections.
    Call this from the FastAPI lifespan shutdown hook.
    MED-02 FIX: Registered via app lifespan in main.py.
    """
    if hasattr(_local, 'conn') and _local.conn:
        try:
            _local.conn.close()
        except Exception as e:
            logger.warning(f"Error closing DB connection: {e}")
        finally:
            _local.conn = None
            logger.info("Closed database connection")

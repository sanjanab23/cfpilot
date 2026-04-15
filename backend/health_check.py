"""
Health check and monitoring endpoints for production observability.
"""
import os
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from db_utils import get_db_connection

# Reuse the same bearer auth from main.py to restrict /metrics
# Imported lazily here to avoid circular imports
def _require_admin():
    """Dependency that validates JWT and checks for admin role."""
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi import Request
    from auth_utils import decode_access_token

    bearer = HTTPBearer(auto_error=False)

    async def _inner(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        if not credentials or not credentials.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing",
                headers={"WWW-Authenticate": "Bearer"},
            )
        payload = decode_access_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        if payload.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )
        return payload.get("sub")

    return _inner

logger = logging.getLogger(__name__)

router = APIRouter()

# MED-01 FIX: /health is defined ONLY here (removed from main.py inline definition)
@router.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring systems.
    Returns 200 OK if service is healthy.
    """
    return {
        "status": "healthy",
        "service": "cf-pilot",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check including database connectivity.
    CRIT-04 FIX: Raw exception messages are never returned in the response body.
    """
    db_status = "healthy"
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
    except Exception as e:
        # Log the real error server-side; return generic status to caller
        logger.error(f"DB health check failed: {e}")
        db_status = "unhealthy"  # sanitized — no str(e) in response

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "service": "cf-pilot",
        "version": "1.0.0",
        "components": {
            "database": db_status,
            "api": "healthy"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/metrics")
async def metrics(admin: str = Depends(_require_admin())):
    """
    Basic metrics endpoint for monitoring.
    CRIT-03 FIX: Requires a valid admin JWT. Internal errors are sanitized.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM chat_history")
            message_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM user_contexts")
            context_count = cursor.fetchone()[0]

        return {
            "users_total": user_count,
            "messages_total": message_count,
            "contexts_total": context_count,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Metrics query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

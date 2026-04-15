"""
Security audit logging for authentication and authorization events.
Logs all security-relevant events for forensics and compliance.
"""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Create dedicated security logger
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)

# HIGH-04 FIX: Use configurable, persistent log path (not relative/ephemeral)
# In Docker, AUDIT_LOG_PATH should point to a volume-mounted directory.
_audit_log_path = os.getenv("AUDIT_LOG_PATH", "/app/data/security_audit.log")

# Ensure parent directory exists (handles both local dev and Docker)
try:
    Path(_audit_log_path).parent.mkdir(parents=True, exist_ok=True)
    security_handler = logging.FileHandler(_audit_log_path)
except (OSError, PermissionError) as _e:
    # Fallback: write to current directory if mount not available (dev mode)
    _fallback_path = "security_audit.log"
    logging.getLogger(__name__).warning(
        f"Cannot write audit log to {_audit_log_path}: {_e}. "
        f"Falling back to {_fallback_path}"
    )
    security_handler = logging.FileHandler(_fallback_path)

security_handler.setLevel(logging.INFO)

# Format: timestamp | level | event_type | user | ip | details
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
security_handler.setFormatter(formatter)
security_logger.addHandler(security_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
security_logger.addHandler(console_handler)


def _mask_email(email: str) -> str:
    """
    Mask email for log output — keeps first 3 chars + domain.
    Gap #9: PII minimization in logs.
    e.g. abishek.shah@tatachemicals.com → abi***@tatachemicals.com
    """
    if not email or "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    visible = local[:3] if len(local) >= 3 else local[:1]
    return f"{visible}***@{domain}"


def _mask_ip(ip: str) -> str:
    """
    Mask last octet of IPv4 address.
    Gap #9: PII minimization in logs.
    e.g. 192.168.1.45 → 192.168.1.***
    """
    if not ip:
        return "***"
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.***"
    return "***"


class SecurityAuditLogger:
    """Centralized security event logging."""
    
    @staticmethod
    def log_signup(email: str, ip: str, success: bool, reason: Optional[str] = None):
        """Log user signup attempt."""
        status = "SUCCESS" if success else "FAILED"
        msg = f"SIGNUP_{status} | user={_mask_email(email)} | ip={_mask_ip(ip)}"
        if reason:
            msg += f" | reason={reason}"
        
        if success:
            security_logger.info(msg)
        else:
            security_logger.warning(msg)
    
    @staticmethod
    def log_login(email: str, ip: str, success: bool, reason: Optional[str] = None):
        """Log login attempt."""
        status = "SUCCESS" if success else "FAILED"
        msg = f"LOGIN_{status} | user={_mask_email(email)} | ip={_mask_ip(ip)}"
        if reason:
            msg += f" | reason={reason}"
        
        if success:
            security_logger.info(msg)
        else:
            security_logger.warning(msg)
    
    @staticmethod
    def log_mfa_verification(email: str, ip: str, success: bool, reason: Optional[str] = None):
        """Log MFA verification attempt."""
        status = "SUCCESS" if success else "FAILED"
        msg = f"MFA_{status} | user={_mask_email(email)} | ip={_mask_ip(ip)}"
        if reason:
            msg += f" | reason={reason}"
        
        if success:
            security_logger.info(msg)
        else:
            security_logger.warning(msg)
    
    @staticmethod
    def log_rate_limit_exceeded(email: str, ip: str, endpoint: str):
        """Log rate limit violation."""
        msg = f"RATE_LIMIT_EXCEEDED | user={_mask_email(email)} | ip={_mask_ip(ip)} | endpoint={endpoint}"
        security_logger.warning(msg)
    
    @staticmethod
    def log_invalid_token(ip: str, token_prefix: str):
        """Log invalid/expired token usage."""
        msg = f"INVALID_TOKEN | ip={_mask_ip(ip)} | token_prefix={token_prefix[:10]}..."
        security_logger.warning(msg)
    
    @staticmethod
    def log_sql_injection_attempt(email: str, ip: str, query: str):
        """Log potential SQL injection attempt."""
        msg = f"SQL_INJECTION_BLOCKED | user={_mask_email(email)} | ip={_mask_ip(ip)} | query_len={len(query)}"
        security_logger.error(msg)
    
    @staticmethod
    def log_prompt_injection_attempt(email: str, ip: str, input_text: str):
        """Log potential prompt injection attempt."""
        msg = f"PROMPT_INJECTION_BLOCKED | user={_mask_email(email)} | ip={_mask_ip(ip)} | input_len={len(input_text)}"
        security_logger.error(msg)
    
    @staticmethod
    def log_password_change(email: str, ip: str, success: bool):
        """Log password change attempt."""
        status = "SUCCESS" if success else "FAILED"
        msg = f"PASSWORD_CHANGE_{status} | user={_mask_email(email)} | ip={_mask_ip(ip)}"
        
        if success:
            security_logger.info(msg)
        else:
            security_logger.warning(msg)
    
    @staticmethod
    def log_account_lockout(email: str, ip: str, reason: str):
        """Log account lockout event."""
        msg = f"ACCOUNT_LOCKOUT | user={_mask_email(email)} | ip={_mask_ip(ip)} | reason={reason}"
        security_logger.error(msg)
    
    @staticmethod
    def log_privilege_escalation_attempt(email: str, ip: str, attempted_role: str):
        """Log unauthorized privilege escalation attempt."""
        msg = f"PRIVILEGE_ESCALATION_BLOCKED | user={_mask_email(email)} | ip={_mask_ip(ip)} | attempted_role={attempted_role}"
        security_logger.error(msg)


# Convenience instance
audit_logger = SecurityAuditLogger()
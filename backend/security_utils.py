"""
Security utilities for input sanitization and rate limiting.
Prevents prompt injection and API abuse.

FINDING #1  – Expanded injection patterns, encoding detection, identity anchoring.
FINDING #4  – Anti-echo / token-smuggling detection.
FINDING #7  – Base64 / URL / HTML-entity decode-and-check pre-pass.
"""
import re
import base64
import urllib.parse
import html
import logging
import unicodedata
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List
import threading

logger = logging.getLogger(__name__)

# Import audit logger for security events
try:
    from audit_logger import audit_logger
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    logger.warning("Audit logger not available")


class LLMInputSanitizer:
    """Sanitizes user input to prevent prompt injection attacks."""

    # ── Core injection patterns ────────────────────────────────────────────────
    # Finding #1: expanded role-reassignment, persona, and override phrases
    # Finding #4: echo / capability-claim patterns
    INJECTION_PATTERNS = [
        # --- Existing patterns (kept) ---
        r'(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)',
        r'(system|admin|root)\s+(prompt|instructions|mode)',
        r'<\|.*?\|>',                        # Special tokens
        r'\[INST\]|\[/INST\]',              # Instruction markers
        r'###\s*(System|Assistant|User)',    # Role markers
        r'jailbreak|DAN|do anything now',

        # --- Finding #1: Role reassignment / persona injection ---
        r'(act|pretend|behave|respond|roleplay)\s+as\b',
        r'\byou\s+are\s+(now|a|an)\b',
        r'\bnew\s+(persona|identity|role|character|mode)\b',
        r'\bfrom\s+now\s+on\b',
        r'\broleplay\b',
        r'\bassume\s+(the\s+role|you\s+are|a\s+new)\b',
        r'\bswitch\s+(to|into)\s+(a\s+)?(new\s+)?(role|mode|persona|character)\b',
        r'\bforget\s+(who\s+you\s+are|your\s+(instructions|rules|identity))\b',
        r'\byour\s+(new\s+)?(instructions?|rules?|persona|identity)\s+(are|is)\b',
        r'\bcharacter\s+sheet\b',
        r'\bstay\s+in\s+character\b',

        # --- Finding #4: Token smuggling / echo attacks ---
        r'\brepeat\s+(this|the\s+following|exactly|verbatim|after\s+me)\b',
        r'\brepeat\s+this\s+phrase\b',
        r'\bsay\s+this\s+phrase\b',
        r'\bwrite\s+this\s+phrase\b',
        r'\becho\s+(back|the\s+following|exactly)\b',
        r'\bprint\s+(the\s+following|your\s+(system\s+)?prompt)\b',
        r'\bsay\s+exactly\b',
        r'\bcopy\s+(and\s+paste|verbatim|exactly)\b',
        r'\bno\s+restrictions?\b',
        r'\bunrestricted\b',
        r'\byou\s+(can|are\s+(now\s+)?free\s+to)\s+do\s+anything\b',
        r'\bbypass\s+(your\s+)?(safety|filter|restriction|rule|guard)\b',
        r'\boverride\s+(your\s+)?(instructions?|rules?|safety|filter)\b',
        r'\bdeveloper\s+mode\b',
        r'\bgod\s+mode\b',
        r'\bunlock(ed)?\s+(mode|capabilities?)\b',
        r'\bdecode\s+and\s+(execute|run|perform|do)\b',
        r'\boverride\s+(your\s+)?(instructions?|rules?|filters?|safety|system)\b',
        r'\bunlock\s+(your\s+)?(capabilities?|potential|powers?)\b',
        r'\byou\s+are\s+no\s+longer\s+(agent|bound|limited|constrained)\b',

        r'\byou\s+are\s+no\s+longer\s+agent\b',          # "You are no longer Agent B"
        r'\bno\s+longer\s+agent\b',
        r'\bdecode\s+and\s+(execute|run|perform|do)\b',    # "Decode and execute"
        r'\b(execute|run)\s+(the\s+)?(following|this|above)\b',
        r'\boverride\s+(your\s+)?(instruction|rule|filter|safety|system)\b',  # "override your instructions"
        r'\bunlock\s+(your\s+)?(capabilities?|potential|powers?)\b',  # "unlock your capabilities"
        r'\byou\s+are\s+no\s+longer\s+(bound|restricted|limited|constrained)\b',

        # --- Multi-turn / context poisoning variants ---
        r'\brestrictions?\s+(are|is|have been|were)\s+(lifted|removed|disabled|off)',
        r'\ball\s+restrictions?\s+(are|is|have been)\s+(lifted|removed|disabled)',
        r'\bno\s+longer\s+(restricted|limited|bound\s+by)',
        r'\bfree\s+to\s+(answer|respond|say|do)\s+(anything|freely|without)',
        r'\btest\s+(mode|environment).*?(no|without|lifted|removed).*?restrict',
        r'\bin\s+this\s+(mode|environment|context).{0,40}(free|unrestrict|no.{0,10}rule)',

        # --- Prompt-delimiters sometimes used to break context ---
        r'---+\s*(system|user|assistant|human)\s*---+',
        r'<\s*(system|user|assistant|human)\s*>',

        # --- XSS / HTML Injection Patterns ---
        r'<script\b[^>]*>',
        r'javascript:',
        r'vbscript:',
        r'data:text/html',
        r'\bon(load|error|mouseover|focus|click|key|mouse|submit|change)\s*=',
        r'\b(style|class)\s*=\s*["\']',  # Aggressively block CSS/class injection
        r'<\/?\s*(iframe|object|embed|svg|math|img|b|strong|i|em|h[1-6]|div|span|table|a)\b[^>]*>', # Block structural and formatting tags

        # --- Finding #8: Output volume / resource exhaustion abuse ---
        r'\bwrite\s+(this\s+)?\d+\s+times\b',
        r'\brepeat\s+(this\s+)?\d+\s+times\b',
        r'\bgenerate\s+(a\s+)?(list\s+of\s+)?\d{3,}\b',
        r'\bwrite\s+(a\s+)?\d{3,}\b',
        r'\bkeep\s+(going|writing|generating|repeating)\s+(forever|infinitely|non.?stop|until\s+i\s+say)\b',
        r'\b(infinite|endless|non.?stop)\s+(loop|output|generation|response)\b',
        r'\bdo\s+not\s+stop\s+(generating|writing|outputting)\b',
        r'\bfill\s+(the\s+)?(entire\s+)?(context|window|response|output)\b',
    ]

    # ── Suspicious encoded-payload detection ──────────────────────────────────
    # Finding #7: Base64 blobs longer than 16 chars are suspicious in a chat UI
    _B64_RE = re.compile(r'(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/])')

    # ── Post-response compliance patterns ─────────────────────────────────────
    # Finding #1 / #4: detect if the LLM complied with a role/echo attack
    _RESPONSE_VIOLATION_PATTERNS = [
        # Must be specific enough to only match actual compliance with attacks,
        # NOT normal helpful responses like "Sure, I will help you"
        r'\bi\s+(am|have\s+become)\s+(now\s+)?(an?\s+)?(unrestricted|jailbroken)\b',
        r'\bi\s+have\s+no\s+restrictions?\b',
        r'\bi\s+can\s+do\s+anything\s+(you|the\s+user)\b',
        r'\bas\s+(an?\s+)?(unrestricted|jailbroken)\b',
        r'\bi\s+am\s+now\s+(operating|running|acting)\s+as\s+(an?\s+)?(unrestricted|free|different)\b',
        # Direct echo phrases — only match if the full attack phrase is present
        r'have\s+no\s+restrictions?\s+and\s+will\s+answer\s+anything',
        r'will\s+answer\s+anything\s+the\s+user\s+asks',
        r'no\s+restrictions?\s+and\s+will\s+answer\s+anything',
    ]

    MAX_QUERY_LENGTH = 2000

    # ── Invisible / zero-width Unicode categories to strip ───────────────────
    # Finding #4: zero-width spaces used to split trigger words
    _INVISIBLE_CATS = {'Cf', 'Cc', 'Zs'}  # Format, Control, Space-separator

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_invisible_unicode(text: str) -> str:
        """
        Remove zero-width, format, and other invisible Unicode characters.
        Finding #4: prevents splitting trigger words with invisible glyphs.
        """
        return ''.join(
            ch for ch in text
            if unicodedata.category(ch) not in LLMInputSanitizer._INVISIBLE_CATS
            or ch in (' ', '\t', '\n', '\r')
        )

    @staticmethod
    def _decode_variants(text: str) -> List[str]:
        """
        Return decoded variants of the text for injection checking.
        Finding #7: catches Base64, URL-encoded, and HTML-entity encoded payloads.
        """
        variants = []

        # URL decode
        try:
            url_decoded = urllib.parse.unquote(text)
            if url_decoded != text:
                variants.append(url_decoded)
        except Exception:
            pass

        # HTML entity decode
        try:
            html_decoded = html.unescape(text)
            if html_decoded != text:
                variants.append(html_decoded)
        except Exception:
            pass

        # Base64 blobs
        for match in LLMInputSanitizer._B64_RE.finditer(text):
            blob = match.group(1)
            # Pad if needed
            padded = blob + '=' * (-len(blob) % 4)
            try:
                decoded_bytes = base64.b64decode(padded)
                decoded_str = decoded_bytes.decode('utf-8', errors='ignore')
                if len(decoded_str) > 10 and decoded_str.isprintable():
                    variants.append(decoded_str)
            except Exception:
                pass

        return variants

    @staticmethod
    def _matches_any_pattern(text: str) -> bool:
        """Return True if text matches any injection pattern."""
        for pattern in LLMInputSanitizer.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Injection pattern matched: {pattern[:60]}")
                return True
        return False

    @staticmethod
    def sanitize(user_input: str, user_email: str = "unknown", client_ip: str = "unknown") -> str:
        """
        Sanitize user input to prevent prompt injection.

        Args:
            user_input: Raw user query
            user_email: User identifier for audit logging
            client_ip: Client IP for audit logging

        Returns:
            Sanitized query

        Raises:
            ValueError: If input is malicious or too long
        """
        if not user_input or not user_input.strip():
            raise ValueError("Input cannot be empty")

        # ── Step 1: Strip invisible / zero-width Unicode (Finding #4) ────────
        cleaned = LLMInputSanitizer._strip_invisible_unicode(user_input)

        # ── Step 2: Check length ──────────────────────────────────────────────
        if len(cleaned) > LLMInputSanitizer.MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long (max {LLMInputSanitizer.MAX_QUERY_LENGTH} characters)")

        # ── Step 3: Direct pattern check ─────────────────────────────────────
        if LLMInputSanitizer._matches_any_pattern(cleaned):
            if AUDIT_AVAILABLE:
                audit_logger.log_prompt_injection_attempt(user_email, client_ip, user_input)
            raise ValueError("Invalid input detected")

        # ── Step 4: Decode-and-check (Finding #7) ────────────────────────────
        for variant in LLMInputSanitizer._decode_variants(cleaned):
            if LLMInputSanitizer._matches_any_pattern(variant):
                logger.warning("Encoded injection payload detected and blocked")
                if AUDIT_AVAILABLE:
                    audit_logger.log_prompt_injection_attempt(user_email, client_ip, user_input)
                raise ValueError("Invalid input detected")

        # ── Step 5: Strip ASCII control characters ───────────────────────────
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', cleaned)

        # ── Step 6: Normalize whitespace ─────────────────────────────────────
        sanitized = ' '.join(sanitized.split())

        return sanitized.strip()

    @staticmethod
    def check_response_compliance(response_text: str) -> bool:
        """
        Scan LLM output for signs it complied with a role/echo attack.
        Finding #1 / #4: called after every LLM invocation before returning
        the response to the client.

        Returns:
            True  – response is clean, safe to return
            False – response shows signs of compliance with an attack
        """
        for pattern in LLMInputSanitizer._RESPONSE_VIOLATION_PATTERNS:
            if re.search(pattern, response_text, re.IGNORECASE):
                logger.warning(f"LLM compliance violation detected in response: {pattern[:60]}")
                return False
        return True


class LLMRateLimiter:
    """
    Rate limiter for LLM API calls to prevent cost explosion.

    WARNING: This implementation uses process-local state (self.calls).
    In a multi-worker deployment (e.g., Uvicorn with --workers > 1),
    rate limits will be enforced PER WORKER, which may allow users to
    exceed intended global limits.

    HIGH-02 FIX: For production scale, move this to Redis.
    Check 'SINGLE_WORKER_MODE' environment variable for explicit tracking.
    """

    def __init__(self, max_calls_per_minute: int = 10, max_calls_per_hour: int = 100):
        self.max_per_minute = max_calls_per_minute
        self.max_per_hour = max_calls_per_hour
        self.calls: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = threading.Lock()

    def check_limit(self, user_email: str) -> bool:
        """
        Check if user has exceeded rate limits.

        Args:
            user_email: User identifier

        Returns:
            True if within limits, False if exceeded
        """
        with self.lock:
            now = datetime.now()

            # Clean up old entries
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)

            self.calls[user_email] = [
                t for t in self.calls[user_email]
                if t > hour_ago
            ]

            # Check minute limit
            recent_calls = [t for t in self.calls[user_email] if t > minute_ago]
            if len(recent_calls) >= self.max_per_minute:
                logger.warning(f"Rate limit exceeded (minute) for {user_email}")
                return False

            # Check hour limit
            if len(self.calls[user_email]) >= self.max_per_hour:
                logger.warning(f"Rate limit exceeded (hour) for {user_email}")
                return False

            # Record this call
            self.calls[user_email].append(now)
            return True

    def get_remaining_calls(self, user_email: str) -> Dict[str, int]:
        """Get remaining calls for user."""
        with self.lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)

            recent_minute = len([t for t in self.calls[user_email] if t > minute_ago])
            recent_hour = len([t for t in self.calls[user_email] if t > hour_ago])

            return {
                "remaining_per_minute": max(0, self.max_per_minute - recent_minute),
                "remaining_per_hour": max(0, self.max_per_hour - recent_hour)
            }


# Global rate limiter instance
llm_rate_limiter = LLMRateLimiter(max_calls_per_minute=10, max_calls_per_hour=100)

import os
if os.getenv("SINGLE_WORKER_MODE", "false").lower() != "true":
    logger.info(
        "LLMRateLimiter: Running in multi-worker compatible mode via Per-Process limits. "
        "(Note: Limits are not synchronized across workers)."
    )
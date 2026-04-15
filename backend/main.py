import sqlite3
import os
import io
import base64
import logging
import json
import time
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pyotp
import qrcode
import re
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
from fastapi import FastAPI, Request, HTTPException, status, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, field_validator
import uvicorn
from dotenv import load_dotenv


# --- SECURITY IMPORTS ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
# HIGH-01 FIX: import pre-auth token helpers
from auth_utils import (
    get_password_hash, verify_password, create_access_token, decode_access_token,
    create_pre_auth_token, decode_pre_auth_token
)
from security_utils import llm_rate_limiter, LLMInputSanitizer
# --- AI IMPORTS (Global Scope) ---
from brain import app_brain

# --- DATABASE AND SECURITY IMPORTS ---
from db_utils import get_db_connection, execute_query, close_all_connections
from crypto_utils import encrypt_mfa_secret, decrypt_mfa_secret
from audit_logger import audit_logger
from health_check import router as health_router

load_dotenv()

# --- ACCOUNT LOCKOUT CONFIGURATION (Fix CF-04) ---
MAX_FAILED_ATTEMPTS = 5          # Lock after 5 consecutive failures
LOCKOUT_DURATION_MINUTES = 15    # Lock for 15 minutes

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DB_NAME = os.getenv("DB_PATH", "cfpilot.db")

# 1. RATE LIMITING SETUP
def get_real_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host

limiter = Limiter(key_func=get_real_ip)


# MED-02 FIX: Lifespan hook ensures DB connections are gracefully closed on shutdown
def _run_retention_purge() -> None:
    """
    Delete records older than configured retention periods.
    Gap #7: Auto-deletion for chat, context, logs and PPT data.
    Runs once at startup and then every 24 hours.
    """
    import threading
    def purge():
        while True:
            try:
                with get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "DELETE FROM chat_history WHERE timestamp < datetime('now', ?)",
                        (f"-{RETENTION_DAYS_CHAT} days",)
                    )
                    cur.execute(
                        "DELETE FROM user_contexts WHERE updated_at < datetime('now', ?)",
                        (f"-{RETENTION_DAYS_CONTEXT} days",)
                    )
                    cur.execute(
                        "DELETE FROM agent_logs WHERE timestamp < datetime('now', ?)",
                        (f"-{RETENTION_DAYS_LOGS} days",)
                    )
                    cur.execute(
                        "DELETE FROM ppt_history WHERE timestamp < datetime('now', ?)",
                        (f"-{RETENTION_DAYS_PPT} days",)
                    )
                    logger.info(
                        f"Retention purge complete — chat>{RETENTION_DAYS_CHAT}d, "
                        f"context>{RETENTION_DAYS_CONTEXT}d, "
                        f"logs>{RETENTION_DAYS_LOGS}d, ppt>{RETENTION_DAYS_PPT}d"
                    )
            except Exception as e:
                logger.error(f"Retention purge failed: {e}")
            import time
            time.sleep(86400)  # Run every 24 hours

    t = threading.Thread(target=purge, daemon=True)
    t.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    logger.info("Application starting up")
    _run_retention_purge()  # Gap #7: start background retention purge
    yield
    logger.info("Application shutting down — closing DB connections")
    close_all_connections()

# Finding #13: disable OpenAPI/Swagger docs in production
_ENV = os.getenv("ENVIRONMENT", "production").lower()
app = FastAPI(
    title="CFPilot API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs"        if _ENV == "development" else None,
    redoc_url="/redoc"      if _ENV == "development" else None,
    openapi_url="/openapi.json" if _ENV == "development" else None,
)

# JWT Bearer scheme - extracts token from Authorization: Bearer <token> header
bearer_scheme = HTTPBearer(auto_error=False)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Finding #10: catch ALL unhandled exceptions — never leak stack traces to client
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."}
    )

# 2. STRICT CORS SETUP WITH VALIDATION
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

# Fallback for development only
if not allowed_origins:
    logger.warning("ALLOWED_ORIGINS not set - using permissive CORS for development")
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Only allows trusted domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- SECURITY HEADERS MIDDLEWARE (Fix CF-02) ---
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects HTTP security headers into every response.
    Covers: X-Frame-Options, HSTS, CSP, X-Content-Type-Options,
            Referrer-Policy, X-XSS-Protection, Permissions-Policy
    """
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Prevent clickjacking
        # Allow framing from same origin and Tableau for extension support
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        # Stop MIME-type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Control referrer info sent to third parties
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Enforce HTTPS (30-day cache, includes subdomains)
        response.headers["Strict-Transport-Security"] = "max-age=2592000; includeSubDomains"
        # XSS reflection mitigation for older browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Disable browser features not used by the app
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(self), geolocation=(), payment=(), usb=()"
        )
        # Content Security Policy - tuned to CDNs used in index.html and Tableau extension support
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
              "https://code.jquery.com "
              "https://cdn.jsdelivr.net "
              "https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' "
              "https://fonts.googleapis.com "
              "https://cdnjs.cloudflare.com; "
            "font-src 'self' data: "
              "https://fonts.gstatic.com "
              "https://cdnjs.cloudflare.com; "
            "connect-src 'self' https://openrouter.ai https://cdn.jsdelivr.net https://cdnjs.cloudflare.com http://localhost:* http://127.0.0.1:*;"
            "img-src 'self' data:; "
            "frame-ancestors 'self' https://*.tableau.com https://*.tableauusercontent.com http://localhost:* http://127.0.0.1:*;"
        )
        return response

app.add_middleware(SecurityHeadersMiddleware)

# --- SERVE FRONTEND (STATIC FILES) ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# Mount the frontend directory
# Logic: If running in Docker, frontend might be at /app/frontend. 
# If local, it's at ../frontend relative to this file.
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "../frontend")
if not os.path.exists(frontend_dir):
    # Fallback for Docker structure if flat
    frontend_dir = os.path.join(current_dir, "frontend")

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    logger.info(f"Serving static files from: {frontend_dir}")
else:
    logger.warning(f"Frontend directory not found at {frontend_dir}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main application entry point."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>CF Pilot Backend Running</h1><p>Frontend not found.</p>"

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Silence favicon 404 noise."""
    return FileResponse(os.path.join(frontend_dir, "favicon.ico")) if os.path.exists(os.path.join(frontend_dir, "favicon.ico")) else HTMLResponse("")

# --- DATABASE HELPER FUNCTIONS ---
DEFAULT_CONTEXT = "No dashboard data scanned yet. Please scan the dashboard first using the sync button."

def get_user_context(email: str) -> str:
    """Get dashboard context for a specific user — decrypts at read. Gap #8."""
    from crypto_utils import decrypt_value
    row = execute_query(
        "SELECT context FROM user_contexts WHERE email=?",
        (email,),
        fetch_one=True
    )
    if not row:
        return DEFAULT_CONTEXT
    try:
        return decrypt_value(row[0])
    except Exception:
        return row[0]  # Legacy unencrypted row — return as-is

def set_user_context(email: str, context: str) -> None:
    """Set dashboard context for a specific user in database — encrypted at rest. Gap #8."""
    from crypto_utils import encrypt_value
    import re as _re

    # Gap #4: sanitize before storing — strip SF field names, emails, long numbers
    context = _re.sub('\\b\\w+__[cr]\\b', '[field]', context, flags=_re.IGNORECASE)
    context = _re.sub('[\\w.+-]+@[\\w.-]+\\.[\\w]+', '[email]', context)
    context = _re.sub('\\b[0-9]{10,}\\b', '[number]', context)
    if len(context) > 2000:
        context = context[:2000] + '...[truncated]'

    encrypted_context = encrypt_value(context) if context else context

    # Upsert: update if exists, insert if not
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO user_contexts (email, context, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (email, encrypted_context)
        )

# --- DATABASE INIT ---
def init_db() -> None:
    """Initialize the SQLite database with required tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Updated Users Table with Role
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                name TEXT,
                company TEXT,
                mfa_secret TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                mfa_enabled INTEGER DEFAULT 0,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until DATETIME DEFAULT NULL
            )
        ''')
        # Migration: add lockout columns if upgrading from older schema
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0")
        except Exception:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN locked_until DATETIME DEFAULT NULL")
        except Exception:
            pass  # Column already exists
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                chart_data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # New table for user contexts (fixes race condition)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_contexts (
                email TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            )
        ''')

        # Fix CF-05: TOTP replay prevention — stores used codes until their window expires
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS used_totp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                code TEXT NOT NULL,
                used_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(email, code)
            )
        ''')

        # MED-03 FIX: Index on used_at for efficient background cleanup queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_used_totp_used_at
            ON used_totp_codes(used_at)
        ''')

        # Password reset tokens table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE,
                expires_at DATETIME NOT NULL,
                used INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # NEW: PPT History table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ppt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                prompt TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # NEW: Agent Logs table for detailed tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                action_type TEXT NOT NULL, -- e.g., 'CHAT_LLM', 'PPT_PLAN', 'SQL_QUERY'
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

init_db()

# MED-05 FIX: Verify PPT template exists on startup to prevent late-stage failures
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "Tata_Chemicals_PPT_Template_FY_25.pptx")
if not os.path.exists(TEMPLATE_PATH):
    logger.error(f"CRITICAL: PPT Template not found at {TEMPLATE_PATH}")
    # In production, we should raise SystemExit, but for dev we'll just log loudly
    if os.getenv("ENV") == "production":
        raise RuntimeError(f"Required file {TEMPLATE_PATH} missing for production.")
else:
    logger.info(f"PPT Template verified at {TEMPLATE_PATH}")

# --- PYDANTIC MODELS WITH ENHANCED VALIDATION ---

# HIGH-03 / LOW-01 FIX: Context size cap constant
MAX_CONTEXT_BYTES = 100_000   # 100 KB max per user context

# Gap #7: Data retention periods (configurable via env vars)
RETENTION_DAYS_CHAT    = int(os.getenv("RETENTION_DAYS_CHAT",    "90"))   # chat_history
RETENTION_DAYS_CONTEXT = int(os.getenv("RETENTION_DAYS_CONTEXT", "30"))   # user_contexts
RETENTION_DAYS_LOGS    = int(os.getenv("RETENTION_DAYS_LOGS",    "30"))   # agent_logs
RETENTION_DAYS_PPT     = int(os.getenv("RETENTION_DAYS_PPT",     "90"))   # ppt_history

class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128, description="Password must be at least 8 characters")
    name: str = Field(..., min_length=1, max_length=100)
    company: str = Field(..., max_length=200)
    

    @field_validator('email')
    @classmethod
    def validate_email_case(cls, v: str) -> str:
        return v.lower()

    @field_validator('company')
    @classmethod
    def validate_company(cls, v: str) -> str:
        allowed_companies = {'tcl', 'tcml', 'tcna', 'tce'}
        if v.lower() not in allowed_companies:
            sorted_companies = sorted(list(allowed_companies))
            raise ValueError(f'Company must be one of: {", ".join(sorted_companies).upper()}')
        return v.lower()
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ResetPasswordMfa(BaseModel):
    email: EmailStr
    mfa_code: str
    new_password: str

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v

class VerifyMFA(BaseModel):
    email: EmailStr
    code: str
    # HIGH-01 FIX: pre_auth_token ties this request to a completed /auth/login
    pre_auth_token: str = Field(..., min_length=1, description="Token issued by /auth/login")

class ChatRequest(BaseModel):
    # LOW-01 FIX: enforce message length limit to prevent unbounded LLM cost
    message: str = Field(..., min_length=1, max_length=2000)
    context: dict = {}

class TokenRequest(BaseModel):
    token: str = ""  # Deprecated: token now read from Authorization header

# --- SECURITY HELPER ---
def get_current_user(token: str) -> str:
    """
    Validates the JWT. If valid, returns the user email.
    If invalid, raises an HTTP 401 Unauthorized error.
    """
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
        )
    return payload.get("sub")

def get_current_user_from_header(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> str:
    """
    Extracts and validates JWT from the Authorization: Bearer <token> header.
    Use as a FastAPI Depends() dependency on protected endpoints.
    Raises HTTP 401 if token is missing or invalid.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Use: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return get_current_user(credentials.credentials)

# --- LOGGING HELPER ---
def log_agent_action(email: str, action_type: str, details: str) -> None:
    """Log an agent action to the database."""
    try:
        execute_query(
            "INSERT INTO agent_logs (email, action_type, details) VALUES (?, ?, ?)",
            (email, action_type, details)
        )
    except Exception as e:
        logger.error(f"Error logging agent action: {e}")

# --- DATABASE HELPER ---
def save_chat_message(email: str, sender: str, message: str, charts: Optional[List[dict]] = None) -> None:
    """Save a chat message to the database — message encrypted at rest. Gap #8."""
    try:
        import json
        from crypto_utils import encrypt_value

        # Gap #8: encrypt message before storing
        encrypted_message = encrypt_value(message) if message else message

        # Serialize charts list to JSON string if present
        chart_json = json.dumps(charts) if charts else None

        execute_query(
            "INSERT INTO chat_history (email, sender, message, chart_data) VALUES (?, ?, ?, ?)",
            (email, sender, encrypted_message, chart_json)
        )
    except Exception as e:
        logger.error(f"Error saving chat: {e}")

# --- HEALTH CHECK ENDPOINT ---
# MED-01 FIX: /health is now exclusively defined in health_check.py (via health_router).
# The inline definition has been removed to eliminate the duplicate-route conflict.
# health_router exposes: /health, /health/detailed, /metrics

# --- API ENDPOINTS ---

# --- EMAIL HELPER ---
def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP. Returns True on success, False on failure."""
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    if not smtp_host or not smtp_user or not smtp_pass:
        logger.warning("SMTP not configured — skipping email send")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_from, to_email, msg.as_string())
        logger.info(f"Email sent to {to_email}: {subject}")
        return True
    except Exception as e:
        logger.error(f"Email send failed to {to_email}: {e}")
        return False


@app.post("/auth/signup")
@limiter.limit("5/minute") # Security: Rate Limit
async def signup(request: Request, user: UserSignup):
    client_ip = request.client.host
    
    # --- STRICT WHITELIST CHECK (FAIL-CLOSED) ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        whitelist_file = os.path.join(current_dir, "authorized_emails.txt")
        
        if not os.path.exists(whitelist_file):
            logger.error(f"CRITICAL: Whitelist file not found at {whitelist_file}")
            return {"status": "error", "message": "Sign-up is currently restricted. Please contact an administrator."}
            
        with open(whitelist_file, "r", encoding="utf-8") as f:
            valid_emails = {line.strip().lower() for line in f if line.strip()}
        
        if user.email.lower().strip() not in valid_emails:
            logger.warning(f"Unauthorized signup attempt: {user.email} from {client_ip}")
            audit_logger.log_signup(user.email, client_ip, False, "Email not in approved list")
            return {"status": "error", "message": "This email is not authorized to sign up. Please contact an administrator."}
            
    except Exception as e:
        logger.error(f"Error during whitelist verification: {e}")
        return {"status": "error", "message": "Could not verify authorization at this time. Please try again later."}
    # --- END STRICT WHITELIST CHECK ---

    row = execute_query(
        "SELECT * FROM users WHERE email=?",
        (user.email,),
        fetch_one=True
    )
    if row:
        audit_logger.log_signup(user.email, client_ip, False, "User already exists")
        # Finding #9: do not confirm whether an email is registered
        return {"status": "error", "message": "If this email is eligible, you will receive setup instructions."}
    
    # Security: Hash Password
    hashed_pw = get_password_hash(user.password)
    mfa_secret = pyotp.random_base32()
    
    # Security: Encrypt MFA secret before storage
    encrypted_mfa = encrypt_mfa_secret(mfa_secret)

    # Auto-assign admin role for the designated admin email
    ADMIN_EMAIL = "shlokm@tatachemicals.com" , "itintern1@tatachemicals.com"
    assigned_role = "admin" if user.email.lower() == ADMIN_EMAIL else "user"
    
    try:
        execute_query(
            "INSERT INTO users (email, password, name, company, mfa_secret, mfa_enabled, role) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (user.email, hashed_pw, user.name, user.company, encrypted_mfa, assigned_role)
        )
        logger.info(f"New user registered: {user.email}")
        audit_logger.log_signup(user.email, client_ip, True)

        # Generate QR Code for Initial Setup
        totp = pyotp.TOTP(mfa_secret)
        uri = totp.provisioning_uri(name=user.email, issuer_name="CF Pilot")
        img = qrcode.make(uri)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        qr_b64 = base64.b64encode(buffered.getvalue()).decode()

        # HIGH-01 FIX: Signup also issues a pre-auth token so the user can
        # immediately complete MFA setup at /auth/verify_mfa.
        pre_auth_token = create_pre_auth_token(user.email)

        return {
            "status": "success", 
            "message": "Account created! Scan the QR code to set up 2FA.",
            "qr_code": qr_b64,
            "pre_auth_token": pre_auth_token
        }
    except Exception as e:
        logger.error(f"Signup error for {user.email}: {e}")
        audit_logger.log_signup(user.email, client_ip, False, str(e))
        return {"status": "error", "message": "An error occurred during signup."}

@app.post("/auth/login")
@limiter.limit("10/minute") # Security: Rate Limit
async def login(request: Request, user_data: UserLogin):
    client_ip = request.client.host
    from datetime import datetime, timezone

    row = execute_query(
        "SELECT password, mfa_secret, failed_login_attempts, locked_until FROM users WHERE email=?",
        (user_data.email,),
        fetch_one=True
    )

    # User not found — return generic message (don't reveal whether email exists)
    if not row:
        audit_logger.log_login(user_data.email, client_ip, False, "User not found")
        return {"status": "error", "message": "Invalid Email or Password"}

    stored_password, encrypted_mfa, failed_attempts, locked_until = row

    # HIGH-06 FIX: Normalize locked_until to UTC-aware datetime regardless of
    # whether SQLite stored it with a '+00:00' suffix (Python isoformat) or
    # without (SQLite CURRENT_TIMESTAMP). Both forms are handled safely.
    if locked_until:
        try:
            lock_dt = datetime.fromisoformat(locked_until)
            if lock_dt.tzinfo is None:
                # Naive string from SQLite — assume UTC
                lock_dt = lock_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            # Unparseable value — treat as expired and reset
            lock_dt = datetime.now(timezone.utc)
        if datetime.now(timezone.utc) < lock_dt:
            remaining = int((lock_dt - datetime.now(timezone.utc)).total_seconds() / 60) + 1
            audit_logger.log_account_lockout(user_data.email, client_ip, f"Account locked, {remaining}min remaining")
            return {
                "status": "error",
                "message": f"Account temporarily locked due to too many failed attempts. Try again in {remaining} minute(s)."
            }
        else:
            # Lock expired — reset counter
            execute_query(
                "UPDATE users SET failed_login_attempts=0, locked_until=NULL WHERE email=?",
                (user_data.email,)
            )
            failed_attempts = 0

    # Security: Verify Password Hash
    if verify_password(user_data.password, stored_password):
        # SUCCESS — reset failed counter
        execute_query(
            "UPDATE users SET failed_login_attempts=0, locked_until=NULL WHERE email=?",
            (user_data.email,)
        )

        try:
            decrypt_mfa_secret(encrypted_mfa)
        except Exception as e:
            logger.error(f"MFA decryption failed for {user_data.email}: {e}")
            audit_logger.log_login(user_data.email, client_ip, False, "MFA decryption error")
            return {"status": "error", "message": "Authentication error. Please contact support."}

        # HIGH-01 FIX: Issue a short-lived pre-auth token that MUST be presented
        # at /auth/verify_mfa. Without it, an attacker cannot reach MFA step 2
        # without first completing a valid password check here.
        pre_auth_token = create_pre_auth_token(user_data.email)

        logger.info(f"Successful primary auth for {user_data.email}")
        audit_logger.log_login(user_data.email, client_ip, True)
        return {
            "status": "success",
            "message": "Credentials Verified. Please enter 2FA code.",
            "mfa_required": True,
            "pre_auth_token": pre_auth_token,
        }

    # FAILURE — increment counter and check threshold
    new_failed = (failed_attempts or 0) + 1
    if new_failed >= MAX_FAILED_ATTEMPTS:
        from datetime import timedelta
        lock_until_dt = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        execute_query(
            "UPDATE users SET failed_login_attempts=?, locked_until=? WHERE email=?",
            (new_failed, lock_until_dt.isoformat(), user_data.email)
        )
        audit_logger.log_account_lockout(
            user_data.email, client_ip,
            f"Locked after {new_failed} failed attempts for {LOCKOUT_DURATION_MINUTES} minutes"
        )
        return {
            "status": "error",
            "message": f"Account locked for {LOCKOUT_DURATION_MINUTES} minutes after {MAX_FAILED_ATTEMPTS} failed attempts."
        }
    else:
        execute_query(
            "UPDATE users SET failed_login_attempts=? WHERE email=?",
            (new_failed, user_data.email)
        )

    logger.warning(f"Failed login attempt for {user_data.email} from {client_ip} ({new_failed}/{MAX_FAILED_ATTEMPTS})")
    audit_logger.log_login(user_data.email, client_ip, False, f"Invalid credentials ({new_failed}/{MAX_FAILED_ATTEMPTS})")
    # Finding #9: never reveal attempt count or lockout logic to the client
    return {"status": "error", "message": "Invalid Email or Password."}


@app.post("/auth/reset-password-mfa")
@limiter.limit("5/minute")
async def reset_password_mfa(request: Request, data: ResetPasswordMfa):
    """Validate MFA code and update password."""
    client_ip = request.client.host
    
    # 1. Verify user exists and get MFA secret
    row = execute_query(
        "SELECT mfa_secret, mfa_enabled FROM users WHERE email=?",
        (data.email.lower(),),
        fetch_one=True
    )
    
    if not row:
        return {"status": "error", "message": "Invalid email or MFA code."}
        
    encrypted_mfa, mfa_enabled = row
    
    if not mfa_enabled:
        return {"status": "error", "message": "MFA is not enabled for this account. Cannot reset password via Authenticator."}
        
    # 2. Replay Protection - check if code was already used
    used = execute_query(
        "SELECT id FROM used_totp_codes WHERE email=? AND code=?",
        (data.email.lower(), data.mfa_code),
        fetch_one=True
    )
    if used:
        return {"status": "error", "message": "This Authenticator code has already been used. Please wait for a new code."}
        
    # 3. Verify MFA code
    try:
        secret = decrypt_mfa_secret(encrypted_mfa)
        totp = pyotp.TOTP(secret)
        if not totp.verify(data.mfa_code, valid_window=1):
            return {"status": "error", "message": "Invalid Authenticator code."}
    except Exception as e:
        logger.error(f"MFA verification failed during password reset for {data.email}: {e}")
        return {"status": "error", "message": "Verification error. Please contact support."}
        
    # 4. Mark code as used
    try:
        execute_query(
            "INSERT INTO used_totp_codes (email, code) VALUES (?, ?)",
            (data.email.lower(), data.mfa_code)
        )
    except sqlite3.IntegrityError:
        return {"status": "error", "message": "This Authenticator code has already been used. Please wait for a new code."}
        
    # 5. Update Password
    hashed_pw = get_password_hash(data.new_password)
    execute_query(
        "UPDATE users SET password=?, failed_login_attempts=0, locked_until=NULL WHERE email=?",
        (hashed_pw, data.email.lower())
    )
    
    logger.info(f"Password reset via MFA successful for {data.email}")
    return {"status": "success", "message": "Password updated successfully! You can now log in."}


# --- ADMIN HELPERS ---
ADMIN_EMAIL = "itintern1@tatachemicals.com"

def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> str:
    """Dependency: only allows the designated admin email."""
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    email = payload.get("sub", "")
    role = payload.get("role", "user")
    if email.lower() != ADMIN_EMAIL and role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return email


@app.get("/admin/users")
@limiter.limit("30/minute")
async def admin_get_users(request: Request, admin: str = Depends(require_admin)):
    """Return list of all registered users with usage stats (admin only)."""
    rows = execute_query(
        """
        SELECT 
            u.email, u.name, u.company, u.role, u.mfa_enabled, u.failed_login_attempts, u.locked_until,
            (SELECT COUNT(*) FROM chat_history ch WHERE ch.email = u.email AND ch.sender = 'user') as chat_count,
            (SELECT COUNT(*) FROM ppt_history ph WHERE ph.email = u.email) as ppt_count
        FROM users u
        ORDER BY u.rowid ASC
        """,
        fetch_all=True
    )
    users = []
    for row in (rows or []):
        email, name, company, role, mfa_enabled, failed_attempts, locked_until, chat_count, ppt_count = row
        users.append({
            "email": email,
            "name": name or "",
            "company": (company or "").upper(),
            "role": role or "user",
            "mfa_enabled": bool(mfa_enabled),
            "failed_attempts": failed_attempts or 0,
            "locked": locked_until is not None,
            "chat_count": chat_count or 0,
            "ppt_count": ppt_count or 0
        })
    return {"status": "success", "users": users, "total": len(users)}


@app.get("/admin/stats")
@limiter.limit("30/minute")
async def admin_get_stats(request: Request, admin: str = Depends(require_admin)):
    """Return system stats (admin only)."""
    total_users = execute_query("SELECT COUNT(*) FROM users", fetch_one=True)
    mfa_enabled_count = execute_query("SELECT COUNT(*) FROM users WHERE mfa_enabled=1", fetch_one=True)
    
    # Detail count of prompts
    chat_prompts = execute_query("SELECT COUNT(*) FROM chat_history WHERE sender='user'", fetch_one=True)
    ppt_generations = execute_query("SELECT COUNT(*) FROM ppt_history", fetch_one=True)
    
    recent_signups = execute_query(
        "SELECT COUNT(*) FROM users WHERE rowid > (SELECT MAX(rowid) - 10 FROM users)",
        fetch_one=True
    )
    return {
        "status": "success",
        "stats": {
            "total_users": total_users[0] if total_users else 0,
            "mfa_enabled": mfa_enabled_count[0] if mfa_enabled_count else 0,
            "chat_prompts": chat_prompts[0] if chat_prompts else 0,
            "ppt_generations": ppt_generations[0] if ppt_generations else 0,
            "recent_signups": recent_signups[0] if recent_signups else 0
        }
    }


@app.get("/admin/logs")
@limiter.limit("30/minute")
async def admin_get_logs(request: Request, admin: str = Depends(require_admin)):
    """Return recent agent logs (admin only)."""
    rows = execute_query(
        "SELECT email, action_type, details, timestamp FROM agent_logs ORDER BY id DESC LIMIT 100",
        fetch_all=True
    )
    logs = []
    for row in (rows or []):
        email, action_type, details, timestamp = row
        logs.append({
            "email": email,
            "action_type": action_type,
            "details": details,
            "timestamp": timestamp
        })
    return {"status": "success", "logs": logs}

@app.post("/auth/verify_mfa")
@limiter.limit("5/minute")
async def verify_mfa(
    request: Request,
    data: VerifyMFA,
    background_tasks: BackgroundTasks
):
    client_ip = request.client.host

    # HIGH-01 FIX: Validate pre_auth_token before touching MFA.
    # This ensures /auth/login (password check) was completed first.
    pre_auth_email = decode_pre_auth_token(data.pre_auth_token)
    if not pre_auth_email or pre_auth_email != data.email:
        audit_logger.log_mfa_verification(
            data.email, client_ip, False, "Invalid or missing pre-auth token"
        )
        return {"status": "error", "message": "Invalid or expired authentication session. Please log in again."}

    row = execute_query(
        "SELECT mfa_secret, name, role FROM users WHERE email=?",
        (data.email,),
        fetch_one=True
    )

    if not row:
        audit_logger.log_mfa_verification(data.email, client_ip, False, "User not found")
        # Finding #9: generic message — do not reveal whether email exists
        return {"status": "error", "message": "Verification failed. Please check your code and try again."}

    encrypted_mfa, name, role = row

    # Decrypt MFA secret
    try:
        mfa_secret = decrypt_mfa_secret(encrypted_mfa)
    except Exception as e:
        logger.error(f"MFA decryption failed for {data.email}: {e}")
        audit_logger.log_mfa_verification(data.email, client_ip, False, "Decryption error")
        return {"status": "error", "message": "Authentication error. Please contact support."}

    totp = pyotp.TOTP(mfa_secret)

    if totp.verify(data.code, valid_window=1):
        # TOTP Replay Protection — check if this code was already used in this window
        already_used = execute_query(
            "SELECT id FROM used_totp_codes WHERE email=? AND code=?",
            (data.email, data.code),
            fetch_one=True
        )
        if already_used:
            audit_logger.log_mfa_verification(data.email, client_ip, False, "TOTP replay attempt blocked")
            logger.warning(f"TOTP replay attempt blocked for {data.email} from {client_ip}")
            return {"status": "error", "message": "Invalid MFA Code"}

        # Mark this code as used
        try:
            execute_query(
                "INSERT INTO used_totp_codes (email, code) VALUES (?, ?)",
                (data.email, data.code)
            )
        except Exception:
            pass  # UNIQUE constraint hit means concurrent replay — safe to reject

        # MED-03 FIX: Move TOTP cleanup to background task (out of hot auth path)
        def _cleanup_totp():
            try:
                execute_query(
                    "DELETE FROM used_totp_codes WHERE used_at < datetime('now', '-90 seconds')"
                )
            except Exception as err:
                logger.warning(f"TOTP cleanup failed: {err}")
        background_tasks.add_task(_cleanup_totp)

        # Security: Generate Secure JWT
        access_token = create_access_token(data={"sub": data.email, "role": role})

        # Confirm MFA Setup
        execute_query(
            "UPDATE users SET mfa_enabled=1 WHERE email=?",
            (data.email,)
        )

        audit_logger.log_mfa_verification(data.email, client_ip, True)
        return {
            "status": "success",
            "token": access_token,
            "username": name
        }
    else:
        audit_logger.log_mfa_verification(data.email, client_ip, False, "Invalid code")
        return {"status": "error", "message": "Invalid MFA Code"}

# --- RATE LIMITING STRATEGY ---
from slowapi.util import get_remote_address

def get_user_email_for_rate_limit(request: Request) -> str:
    """
    Extracts user email from JWT for rate limiting.
    Falls back to IP address if no valid token is present.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        payload = decode_access_token(token)
        if payload and "sub" in payload:
            return payload["sub"]
    return get_remote_address(request)

@app.post("/chat")
@limiter.limit("20/minute;60/day", key_func=get_user_email_for_rate_limit) # Security: Per-user Rate Limit
async def chat_endpoint(
    request: Request,
    data: ChatRequest,
    user_email: str = Depends(get_current_user_from_header)
):
    # Security: JWT validated via Authorization: Bearer header
    
    # Check LLM rate limit
    if not llm_rate_limiter.check_limit(user_email):
        remaining = llm_rate_limiter.get_remaining_calls(user_email)
        return {
            "response": f"⚠️ **Rate Limit Exceeded:** You've reached your query limit. Please wait a moment before trying again. (Resets: {remaining['remaining_per_minute']} calls/min, {remaining['remaining_per_hour']} calls/hour)"
        }

    client_ip = get_real_ip(request)  # Finding #4/#12: needed for audit log on output violation

    # Logic: Standard Chat (all queries go directly to Agent C)

    # MED-04 FIX: Apply LLMInputSanitizer to /chat — previously missing.
    # This blocks prompt injection payloads from reaching app_brain.
    try:
        sanitized_message = LLMInputSanitizer.sanitize(data.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid input. Please revise your message and try again.")

    # Save User Message (use sanitized version)
    save_chat_message(user_email, "user", sanitized_message)
    
    # Execute AI Graph (all queries route directly to Agent C)
    logger.info(f"User Query ({user_email}): {sanitized_message}")
    inputs = {
        "user_query": sanitized_message,
        "dashboard_data": "",  # Not used — all queries go via Agent C
        "messages": [],
        "final_response": "",
        "charts": []
    }
    
    try:
        import traceback
        log_agent_action(user_email, "CHAT_LLM_START", sanitized_message[:200])
        result = app_brain.invoke(inputs)
        final_response = result.get("final_response", "")
        charts = result.get("charts") or []
        logger.info(f"Final Brain Outcome: Response length {len(final_response)}, Charts count {len(charts)}")
        log_agent_action(user_email, "CHAT_LLM_END", f"Response length: {len(final_response)}, Charts: {len(charts)}")

        if not LLMInputSanitizer.check_response_compliance(final_response):
            logger.warning(f"LLM output violation detected for {user_email} — suppressing response")
            audit_logger.log_prompt_injection_attempt(user_email, client_ip, final_response[:200])
            final_response = "⚠️ I'm unable to respond to that request. Please rephrase your question."
            charts = []

        # Finding #8: Post-generation response length guard — prevents resource exhaustion
        # via excessive output even when input patterns are not explicitly matched.
        MAX_RESPONSE_CHARS = 8000  # ~2048 tokens at ~4 chars/token
        if len(final_response) > MAX_RESPONSE_CHARS:
            logger.warning(f"Oversized response ({len(final_response)} chars) for {user_email} — truncating")
            audit_logger.log_event("OUTPUT_THROTTLE", user_email, client_ip, f"Response truncated from {len(final_response)} chars")
            final_response = final_response[:MAX_RESPONSE_CHARS] + "\n\n⚠️ Response truncated to stay within output limits."

    except Exception as e:
        import traceback
        error_str = str(e)
        logger.error(f"Brain Error for user {user_email}: {error_str}\n{traceback.format_exc()}")
        if "400" in error_str or "invalid request" in error_str.lower():
            logger.error("Likely OpenRouter provider/model issue — check Agent C llm_coder config")
        final_response = "⚠️ System Error: Unable to process request."
        charts = []


    # Save Bot Response with charts list
    save_chat_message(user_email, "bot", final_response, charts)
    
    return {
        "response": final_response,
        "charts": charts
    }

@app.post("/chat/history")
@limiter.limit("30/minute")  # Security: Rate Limit added
async def get_chat_history(
    request: Request,
    user_email: str = Depends(get_current_user_from_header)
):
    # Security: JWT validated via Authorization: Bearer header
    
    import json
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # LOW-02 FIX: Paginate history — fetch last 200 messages to avoid O(N) memory spike
        # on large histories. Results are fetched DESC then reversed for chronological order.
        cursor.execute(
            "SELECT sender, message, chart_data, timestamp FROM chat_history "
            "WHERE email=? ORDER BY id DESC LIMIT 200",
            (user_email,)
        )
        rows = cursor.fetchall()
        rows = list(reversed(rows))  # Restore chronological order
    
    from crypto_utils import decrypt_value
    history = []
    for row in rows:
        sender, message, chart_json, timestamp = row

        # Gap #8: decrypt message — fall back to raw text for legacy unencrypted rows
        try:
            message = decrypt_value(message) if message else message
        except Exception:
            pass  # Legacy row stored before encryption was added — use as-is

        # Parse charts list if present
        charts = []
        if chart_json:
            try:
                parsed = json.loads(chart_json)
                # Handle migration from single chart (dict) to list of charts
                if isinstance(parsed, list):
                    charts = parsed
                elif isinstance(parsed, dict):
                    charts = [parsed]
            except:
                pass  # Ignore malformed JSON

        history.append({
            "sender": sender,
            "text": message,
            "charts": charts,
            "timestamp": timestamp
        })
    
    return {"status": "success", "history": history}

@app.post("/chat/save_message")
@limiter.limit("30/minute")
async def save_message_endpoint(
    request: Request,
    user_email: str = Depends(get_current_user_from_header)
):
    """Save an arbitrary bot message to chat history (e.g. PPT download link)."""
    data = await request.json()
    sender = data.get("sender", "bot")
    text = data.get("text", "")
    if sender not in ("user", "bot"):
        sender = "bot"
    if text:
        save_chat_message(user_email, sender, text)
    return {"status": "success"}

@app.post("/chat/clear")
@limiter.limit("10/minute")  
async def clear_chat_history(
    request: Request,
    user_email: str = Depends(get_current_user_from_header)
):
    
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE email=?", (user_email,))
        cursor.execute("DELETE FROM user_contexts WHERE email=?", (user_email,))
    
    logger.info(f"Chat history cleared for user {user_email}")
    return {"status": "success", "message": "History Cleared"}


# ─── DSAR ENDPOINTS (Gap #11: Data Subject Access Request) ───────────────────

@app.get("/user/my-data")
@limiter.limit("5/minute")
async def export_my_data(
    request: Request,
    user_email: str = Depends(get_current_user_from_header)
):
    """
    DSAR export: return all data held for the authenticated user.
    Gap #11: GDPR/DPDPA data subject access request.
    """
    from crypto_utils import decrypt_value
    import json as _json

    data_export = {"email": user_email, "data": {}}

    with get_db_connection() as conn:
        cur = conn.cursor()

        # Chat history (decrypt messages)
        cur.execute(
            "SELECT sender, message, timestamp FROM chat_history WHERE email=? ORDER BY id ASC",
            (user_email,)
        )
        chat_rows = cur.fetchall()
        chat_export = []
        for sender, message, ts in chat_rows:
            try:
                message = decrypt_value(message)
            except Exception:
                pass  # Legacy unencrypted row
            chat_export.append({"sender": sender, "message": message, "timestamp": ts})
        data_export["data"]["chat_history"] = chat_export

        # User context (decrypt)
        cur.execute(
            "SELECT context, updated_at FROM user_contexts WHERE email=?",
            (user_email,)
        )
        ctx_row = cur.fetchone()
        if ctx_row:
            ctx, updated = ctx_row
            try:
                ctx = decrypt_value(ctx)
            except Exception:
                pass
            data_export["data"]["dashboard_context"] = {"context": ctx, "updated_at": updated}

        # PPT history (prompts only — no content)
        cur.execute(
            "SELECT prompt, timestamp FROM ppt_history WHERE email=? ORDER BY id ASC",
            (user_email,)
        )
        data_export["data"]["ppt_history"] = [
            {"prompt": r[0], "timestamp": r[1]} for r in cur.fetchall()
        ]

        # Agent logs
        cur.execute(
            "SELECT action_type, details, timestamp FROM agent_logs WHERE email=? ORDER BY id ASC",
            (user_email,)
        )
        data_export["data"]["activity_logs"] = [
            {"action": r[0], "details": r[1], "timestamp": r[2]} for r in cur.fetchall()
        ]

        # Account info (no password hash, no MFA secret)
        cur.execute(
            "SELECT name, company, role, mfa_enabled FROM users WHERE email=?",
            (user_email,)
        )
        user_row = cur.fetchone()
        if user_row:
            data_export["data"]["account"] = {
                "name": user_row[0], "company": user_row[1], "mfa_enabled": bool(user_row[4])
            }

    audit_logger.log_login(user_email, request.client.host, True, "DSAR export")
    logger.info(f"DSAR export for {user_email}")
    return {"status": "success", "export": data_export}


@app.delete("/user/my-data")
@limiter.limit("3/minute")
async def delete_my_data(
    request: Request,
    user_email: str = Depends(get_current_user_from_header)
):
    """
    DSAR delete: erase all personal data for the authenticated user.
    Keeps the account row but wipes all content and resets context.
    Gap #11: GDPR/DPDPA right to erasure.
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_history WHERE email=?",    (user_email,))
        cur.execute("DELETE FROM user_contexts WHERE email=?",   (user_email,))
        cur.execute("DELETE FROM ppt_history WHERE email=?",     (user_email,))
        cur.execute("DELETE FROM agent_logs WHERE email=?",      (user_email,))
        cur.execute("DELETE FROM used_totp_codes WHERE email=?", (user_email,))

    audit_logger.log_login(user_email, request.client.host, True, "DSAR delete")
    logger.info(f"DSAR deletion complete for {user_email}")
    return {"status": "success", "message": "All your personal data has been deleted."}


# ─── PPT GENERATOR ENDPOINTS ──────────────────────────────────────────────────
import tempfile
import uuid
from fastapi.responses import FileResponse
from ppt_brain import generate_slide_plan
from ppt_generator import generate_pptx

# In-memory store for generated PPTX files (keyed by token)
_ppt_store: dict = {}

class PPTRequest(BaseModel):
    message: str
    context: Optional[str] = None   # optional extra context hint

class PPTEditRequest(BaseModel):
    slides:      list
    edit_index:  int
    edit_prompt: str


# FIX 1: Helper — derive a safe .pptx filename from the cover slide title
def _make_ppt_filename(slides: list) -> str:
    """
    Derive a sanitized .pptx filename from the first cover slide's title.
    Falls back to 'CFPilot_Presentation.pptx' when no title is found.
    Example: "Opportunity Pipeline Summary" -> "Opportunity_Pipeline_Summary.pptx"
    """
    title = ""
    for s in slides:
        if s.get("slide_type") == "cover" and s.get("title"):
            title = s["title"]
            break

    if not title:
        return "CFPilot_Presentation.pptx"

    # Strip characters unsafe in filenames, collapse whitespace to underscores
    safe = re.sub(r"[^\w\s\-]", "", title)        # keep word chars, spaces, hyphens
    safe = re.sub(r"\s+", "_", safe.strip())       # spaces -> underscores
    safe = re.sub(r"_+", "_", safe)                # collapse consecutive underscores
    safe = safe[:80]                               # cap at 80 chars
    return f"{safe}.pptx" if safe else "CFPilot_Presentation.pptx"


@app.post("/ppt/plan")
@limiter.limit("1/minute;10/day", key_func=get_user_email_for_rate_limit)
async def ppt_plan_endpoint(
    request: Request,
    data: PPTRequest,
    user_email: str = Depends(get_current_user_from_header)
):
    """
    Step 1 of PPT flow: Generate slide plan from user prompt.
    Returns JSON slide list for frontend preview.
    """
    if not llm_rate_limiter.check_limit(user_email):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        query = LLMInputSanitizer.sanitize(data.message)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input")

    user_context = get_user_context(user_email)
    logger.info(f"PPT Plan requested by {user_email}: {query[:80]}")

    try:
        # Log PPT generation
        execute_query("INSERT INTO ppt_history (email, prompt) VALUES (?, ?)", (user_email, query))
        log_agent_action(user_email, "PPT_PLAN_START", query[:200])
        
        slides = generate_slide_plan(query, user_context)
        logger.info(f"PPT Plan: {len(slides)} slides generated for {user_email}")
        
        log_agent_action(user_email, "PPT_PLAN_END", f"Generated {len(slides)} slides")
        return {"status": "success", "slides": slides, "count": len(slides)}
    except Exception as e:
        logger.error(f"PPT Plan error for {user_email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate presentation plan")


@app.post("/ppt/edit")
@limiter.limit("1/minute;10/day", key_func=get_user_email_for_rate_limit)
async def ppt_edit_endpoint(
    request: Request,
    data: PPTEditRequest,
    user_email: str = Depends(get_current_user_from_header)
):
    """
    Edit a single slide based on user instruction.
    Returns updated slides list.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage
    import os

    try:
        edit_prompt = LLMInputSanitizer.sanitize(data.edit_prompt)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid edit instruction")

    slides = data.slides
    idx    = data.edit_index

    if idx < 0 or idx >= len(slides):
        raise HTTPException(status_code=400, detail="Invalid slide index")

    slide_to_edit = slides[idx]

    # HIGH-05 FIX: Sanitize all string values in the slide JSON before embedding
    # in the LLM prompt. Prevents prompt injection via a crafted slide payload.
    _MAX_SLIDE_STR_LEN = 500
    def _sanitize_slide_dict(obj, depth=0):
        """Recursively truncate string values; cap depth to prevent DoS."""
        if depth > 5:
            return {}
        if isinstance(obj, dict):
            return {k: _sanitize_slide_dict(v, depth + 1) for k, v in obj.items()
                    if isinstance(k, str) and len(k) <= 100}
        if isinstance(obj, list):
            return [_sanitize_slide_dict(item, depth + 1) for item in obj[:200]]
        if isinstance(obj, str):
            return obj[:_MAX_SLIDE_STR_LEN]
        return obj

    slide_to_edit = _sanitize_slide_dict(slide_to_edit)

    llm_edit = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="xiaomi/mimo-v2-flash",
        temperature=0.2,
        max_retries=3,
        model_kwargs={"extra_body": {"provider": {"zdr": True}}}
    )

    edit_system = f"""
You are a slide editor. Given a slide JSON and an edit instruction, return the UPDATED slide JSON.
Only change what the instruction asks. Keep all other fields unchanged.
Return the JSON object ONLY — no markdown, no explanation.

Current slide:
{json.dumps(slide_to_edit, indent=2)}

Edit instruction: {edit_prompt}

Return the updated slide JSON:
"""
    try:
        resp = llm_edit.invoke([SystemMessage(content=edit_system)])
        import re as _re
        clean = resp.content.strip()
        clean = _re.sub(r"^```(?:json)?\s*", "", clean)
        clean = _re.sub(r"\s*```$", "", clean).strip()
        updated_slide = json.loads(clean)
        slides[idx] = updated_slide
        logger.info(f"Slide {idx} edited for {user_email}")
        return {"status": "success", "slides": slides}
    except Exception as e:
        logger.error(f"PPT edit error: {e}")
        raise HTTPException(status_code=500, detail="Failed to apply edit")


# CRIT-05 FIX: Helper to prune stale temp files (files older than 10 minutes).
# Called as a background task on each download to prevent unbounded disk growth.
PPT_FILE_TTL_SECONDS = 600  # 10 minutes

def _cleanup_ppt_store():
    """Remove stale PPTX entries and temp files from _ppt_store."""
    now = time.time()
    stale_tokens = [
        tok for tok, entry in list(_ppt_store.items())
        if now - entry.get("created_at", 0) > PPT_FILE_TTL_SECONDS
    ]
    for tok in stale_tokens:
        entry = _ppt_store.pop(tok, None)
        if entry:
            try:
                os.unlink(entry["path"])
            except OSError:
                pass
    if stale_tokens:
        logger.info(f"PPT store cleanup: removed {len(stale_tokens)} stale entries")


@app.post("/ppt/download")
@limiter.limit("1/minute;10/day", key_func=get_user_email_for_rate_limit)
async def ppt_download_endpoint(
    request: Request,
    data: dict,
    background_tasks: BackgroundTasks,
    user_email: str = Depends(get_current_user_from_header)
):
    """
    Step 2: Given the final approved slide list, generate and return .pptx file.
    FIX 1: Filename is derived from the cover slide title instead of hard-coded.
    """
    slides = data.get("slides", [])
    if not slides:
        raise HTTPException(status_code=400, detail="No slides provided")

    try:
        pptx_bytes = generate_pptx(slides)

        # FIX 1: Derive filename from cover slide title
        filename = _make_ppt_filename(slides)

        token    = str(uuid.uuid4())
        tmp_path = os.path.join(tempfile.gettempdir(), f"cfpilot_ppt_{token}.pptx")
        with open(tmp_path, "wb") as f:
            f.write(pptx_bytes)

        # CRIT-05 FIX: Store owner + creation time + filename alongside path.
        _ppt_store[token] = {
            "path":       tmp_path,
            "owner":      user_email,
            "created_at": time.time(),
            "filename":   filename,          # FIX 1: stored for retrieval at serve time
        }

        # Schedule stale-file cleanup in background (non-blocking)
        background_tasks.add_task(_cleanup_ppt_store)

        logger.info(f"PPTX generated for {user_email}: {len(pptx_bytes):,} bytes | {filename}")

        return {
            "status":       "success",
            "download_url": f"/ppt/file/{token}",
            "filename":     filename,        # FIX 1: returned to frontend too
        }
    except Exception as e:
        logger.error(f"PPTX generation error for {user_email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate PPTX")


@app.get("/ppt/file/{token}")
async def ppt_file_endpoint(
    token: str,
    background_tasks: BackgroundTasks
):
    """
    Serve a generated PPTX file by token.
    CRIT-02 FIX: Ownership check via token lookup.
    CRIT-05 FIX: Temp file and dict entry deleted after download.
    FIX 1: Filename served to browser matches the presentation title.
    """
    entry = _ppt_store.get(token)
    if not entry:
        raise HTTPException(status_code=404, detail="File not found or expired")

    path = entry["path"]
    if not os.path.exists(path):
        _ppt_store.pop(token, None)
        raise HTTPException(status_code=404, detail="File not found or expired")

    # FIX 1: Read filename from store (set at generation time from cover title)
    filename = entry.get("filename", "CFPilot_Presentation.pptx")

    # File stays available for PPT_FILE_TTL_SECONDS (10 min) — cleaned by _cleanup_ppt_store
    # NOT deleted on first download so user can download multiple times from history

    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename,   
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
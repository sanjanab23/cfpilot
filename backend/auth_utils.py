import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt  # Using pyjwt
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)

# --- CONFIGURATION WITH VALIDATION ---
SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    logger.critical("CRITICAL: JWT_SECRET environment variable is not set!")
    raise RuntimeError(
        "JWT_SECRET must be set in production. Generate with: "
        "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
    )

# Gap #3/#16: JWT key rotation support.
# To rotate:
#   1. Set JWT_SECRET_V2 to a new secret in Azure App Service config.
#   2. New tokens are signed with V2 (kid="v2").
#   3. Old tokens (kid="v1" or no kid) still validate against JWT_SECRET.
#   4. After ACCESS_TOKEN_EXPIRE_MINUTES (default 60 min), all active
#      sessions will have refreshed to V2. You can then retire JWT_SECRET.
SECRET_KEY_V2 = os.getenv("JWT_SECRET_V2")  # Optional — only set during rotation

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# --- PASSWORD HASHING ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Checks if a raw password matches the stored hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Converts a raw password into a secure hash."""
    return pwd_context.hash(password)

# --- JWT TOKEN GENERATION ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a time-limited JSON Web Token.
    Gap #3/#16: Stamps kid claim so rotation can be performed without
    logging out existing users. If JWT_SECRET_V2 is set, new tokens
    use V2 key; old V1 tokens remain valid until they expire.
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    # Use V2 key if available (rotation in progress), otherwise V1
    if SECRET_KEY_V2:
        to_encode["kid"] = "v2"
        signing_key = SECRET_KEY_V2
    else:
        to_encode["kid"] = "v1"
        signing_key = SECRET_KEY

    encoded_jwt = jwt.encode(to_encode, signing_key, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a token. Returns payload or None.
    Gap #3/#16: Tries V2 key first (if set), falls back to V1.
    This allows tokens signed with either key to remain valid
    during a rotation window.
    """
    # Peek at kid without full validation to route to correct key
    keys_to_try = []
    try:
        unverified = jwt.decode(token, options={"verify_signature": False}, algorithms=[ALGORITHM])
        kid = unverified.get("kid", "v1")
        if kid == "v2" and SECRET_KEY_V2:
            keys_to_try = [SECRET_KEY_V2]
        elif kid == "v2" and not SECRET_KEY_V2:
            # V2 token but V2 key not loaded — invalid
            return None
        else:
            # kid=v1 or no kid — try V1, also try V2 as fallback
            keys_to_try = [SECRET_KEY]
            if SECRET_KEY_V2:
                keys_to_try.append(SECRET_KEY_V2)
    except Exception:
        keys_to_try = [SECRET_KEY]

    for key in keys_to_try:
        try:
            payload = jwt.decode(token, key, algorithms=[ALGORITHM])
            return payload
        except jwt.PyJWTError:
            continue
    return None



PRE_AUTH_EXPIRE_MINUTES = 5

def create_pre_auth_token(email: str) -> str:
    """
    Creates a short-lived pre-authentication token.
    Must be presented at /auth/verify_mfa to prove password was verified.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=PRE_AUTH_EXPIRE_MINUTES)
    payload = {"sub": email, "scope": "pre_auth", "exp": expire}
    # Pre-auth tokens always use the current active signing key
    signing_key = SECRET_KEY_V2 if SECRET_KEY_V2 else SECRET_KEY
    return jwt.encode(payload, signing_key, algorithm=ALGORITHM)

def decode_pre_auth_token(token: str) -> Optional[str]:
    """
    Validates a pre-auth token.
    Returns the email (sub) only if the token is valid AND has scope='pre_auth'.
    Returns None on any failure.
    """
    keys = [SECRET_KEY]
    if SECRET_KEY_V2:
        keys.insert(0, SECRET_KEY_V2)
    for key in keys:
        try:
            payload = jwt.decode(token, key, algorithms=[ALGORITHM])
            if payload.get("scope") != "pre_auth":
                return None
            return payload.get("sub")
        except jwt.PyJWTError:
            continue
    return None
"""
Cryptography utilities for encrypting sensitive data.
Used for MFA secrets and other sensitive database fields.
"""
import os
import logging
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Get encryption key from environment
ENCRYPTION_KEY = os.getenv("DB_ENCRYPTION_KEY")

if not ENCRYPTION_KEY:
    # CRITICAL SECURITY FIX: Fail fast if key is missing to prevent data loss
    logger.error("CRITICAL: DB_ENCRYPTION_KEY is missing. Application cannot start safely.")
    raise ValueError("DB_ENCRYPTION_KEY must be set in .env to prevent data loss. Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")

try:
    cipher = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
except Exception as e:
    logger.error(f"Invalid encryption key: {e}")
    raise ValueError("DB_ENCRYPTION_KEY must be a valid Fernet key.")


def encrypt_value(plaintext: str) -> str:
    """
    Encrypt a plaintext string.
    
    Args:
        plaintext: String to encrypt
        
    Returns:
        Encrypted string (base64 encoded)
    """
    if not plaintext:
        raise ValueError("Cannot encrypt empty value")
    
    try:
        encrypted = cipher.encrypt(plaintext.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise


def decrypt_value(encrypted: str) -> str:
    """
    Decrypt an encrypted string.
    
    Args:
        encrypted: Encrypted string (base64 encoded)
        
    Returns:
        Decrypted plaintext string
    """
    if not encrypted:
        raise ValueError("Cannot decrypt empty value")
    
    try:
        decrypted = cipher.decrypt(encrypted.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise


def encrypt_mfa_secret(secret: str) -> str:
    """
    Encrypt an MFA secret for database storage.
    
    Args:
        secret: MFA secret (base32 string)
        
    Returns:
        Encrypted secret
    """
    return encrypt_value(secret)


def decrypt_mfa_secret(encrypted_secret: str) -> str:
    """
    Decrypt an MFA secret from database.
    
    Args:
        encrypted_secret: Encrypted MFA secret
        
    Returns:
        Decrypted MFA secret
    """
    return decrypt_value(encrypted_secret)

"""
Database Migration Script for Phase 2
Migrates existing MFA secrets to encrypted format
"""
import sqlite3
import logging
from crypto_utils import encrypt_mfa_secret

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_NAME = "cfpilot.db"

def migrate_mfa_secrets():
    """Encrypt all existing plaintext MFA secrets."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get all users with MFA secrets
        cursor.execute("SELECT email, mfa_secret FROM users")
        users = cursor.fetchall()
        
        if not users:
            logger.info("No users found to migrate")
            conn.close()
            return
        
        logger.info(f"Found {len(users)} users to migrate")
        
        migrated = 0
        for email, mfa_secret in users:
            try:
                # Check if already encrypted (Fernet tokens start with 'gAAAAA')
                if mfa_secret.startswith('gAAAAA'):
                    logger.info(f"User {email} already has encrypted MFA secret, skipping")
                    continue
                
                # Encrypt the plaintext secret
                encrypted = encrypt_mfa_secret(mfa_secret)
                
                # Update database
                cursor.execute(
                    "UPDATE users SET mfa_secret = ? WHERE email = ?",
                    (encrypted, email)
                )
                
                migrated += 1
                logger.info(f"Migrated MFA secret for {email}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {email}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Migration complete: {migrated}/{len(users)} users migrated")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    print("=" * 60)
    print("MFA SECRET ENCRYPTION MIGRATION")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This will encrypt all MFA secrets in the database")
    print("Make sure DB_ENCRYPTION_KEY is set in your .env file")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() == 'yes':
        migrate_mfa_secrets()
    else:
        print("Migration cancelled")

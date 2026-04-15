import sqlite3
import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

DB_NAME = os.getenv("DB_PATH", "cfpilot.db")
# Ensure absolute path if relative
if not os.path.isabs(DB_NAME):
    # If script run directly, assume DB is in CWD or handle logic
    DB_NAME = os.path.abspath(DB_NAME)

def migrate():
    """Add mfa_enabled column to users table if it doesn't exist."""
    print(f"Migrating database: {DB_NAME}")
    
    if not os.path.exists(DB_NAME):
        logger.error(f"Database file not found at {DB_NAME}")
        return

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "mfa_enabled" not in columns:
            logger.info("Adding mfa_enabled column to users table...")
            # We default to 1 (True) for existing users because they already have secrets generated
            # and we want to enforce MFA for them.
            cursor.execute("ALTER TABLE users ADD COLUMN mfa_enabled INTEGER DEFAULT 1")
            conn.commit()
            logger.info("Migration successful: mfa_enabled column added.")
        else:
            logger.info("Column mfa_enabled already exists. Skipping.")
            
        conn.close()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate()

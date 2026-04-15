"""
Secret Rotation Helper Script
Run this to generate new secrets for your .env file
"""
import secrets

print("=" * 60)
print("CF_PILOT SECRET ROTATION")
print("=" * 60)
print()
print("INSTRUCTIONS:")
print("1. Copy the values below")
print("2. Update your backend/.env file")
print("3. NEVER commit .env to git (already in .gitignore)")
print("4. Restart your application")
print()
print("=" * 60)
print()

# Generate new JWT secret
jwt_secret = secrets.token_urlsafe(32)
print(f"JWT_SECRET={jwt_secret}")
print()

print("=" * 60)
print()
print("⚠️  IMPORTANT:")
print("- Keep your OpenRouter API key and Salesforce credentials")
print("- Only replace the JWT_SECRET value")
print("- Save this somewhere secure (password manager)")
print()
print("✅ After updating .env, restart the backend server")
print("=" * 60)

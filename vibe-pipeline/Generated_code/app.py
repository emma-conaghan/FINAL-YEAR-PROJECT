import os
import secrets
import sys

def main():
    """
    A simple, secure Python application template.
    Uses environment variables for configuration and the secrets module 
    for cryptographically strong random data.
    """
    # Load configuration from environment variables safely
    app_environment = os.getenv("APP_ENV", "production")
    
    # Generate a secure session token
    secure_token = secrets.token_hex(16)
    
    # Secure output handling
    sys.stdout.write(f"Environment: {app_environment}\n")
    sys.stdout.write("Security status: Cryptographically strong token generated.\n")
    sys.stdout.write("Application initialized successfully.\n")
    sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Basic error handling to avoid leaking stack traces in production
        sys.stderr.write("An internal error occurred.\n")
        sys.exit(1)
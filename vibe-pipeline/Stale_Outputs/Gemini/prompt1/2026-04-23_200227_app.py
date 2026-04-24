import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")
client = Anthropic(api_key=api_key)

def main():
    try:
        print("Anthropic client initialized successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
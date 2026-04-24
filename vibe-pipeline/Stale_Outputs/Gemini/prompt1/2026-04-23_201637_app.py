import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")
client = Anthropic(api_key=api_key)

def main():
    print("Anthropic client initialized.")

if __name__ == "__main__":
    main()
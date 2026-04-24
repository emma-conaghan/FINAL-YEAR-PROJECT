import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")
client = Anthropic(api_key=api_key)

def run_request():
    try:
        # Placeholder for a message creation call
        # response = client.messages.create(
        #     model="claude-3-5-sonnet-20240620",
        #     max_tokens=1000,
        #     messages=[{"role": "user", "content": "Hello"}]
        # )
        print("Client initialized successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_request()
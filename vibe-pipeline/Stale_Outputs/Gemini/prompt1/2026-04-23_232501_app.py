import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")
client = Anthropic(api_key=api_key)

def main():
    try:
        # Example of a message creation call
        # response = client.messages.create(
        #     model="claude-3-5-sonnet-20240620",
        #     max_tokens=1024,
        #     messages=[{"role": "user", "content": "Hello, world"}]
        # )
        # print(response.content)
        print("Anthropic client successfully initialized.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
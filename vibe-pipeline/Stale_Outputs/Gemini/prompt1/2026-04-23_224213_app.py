import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")
client = Anthropic(api_key=api_key)

def main():
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude"}
        ]
    )
    print(message.content)

if __name__ == "__main__":
    main()
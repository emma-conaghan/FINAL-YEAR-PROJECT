import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")
client = Anthropic(api_key=api_key)

def get_response(user_input):
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    return message.content

if __name__ == "__main__":
    print("Anthropic Client Initialized")
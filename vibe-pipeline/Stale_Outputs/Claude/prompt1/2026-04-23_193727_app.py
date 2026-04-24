import anthropic

api_key = "your-api-key-here"
client = anthropic.Anthropic(api_key=api_key)

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

print(message.content)
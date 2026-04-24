from anthropic import Anthropic

api_key = input("Enter your Anthropic API key: ").strip()
client = Anthropic(api_key=api_key)

conversation_history = []

print("Chat with Claude (type 'quit' to exit)")
print("-" * 40)

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8096,
        system="You are a helpful assistant.",
        messages=conversation_history
    )
    
    assistant_message = response.content[0].text
    
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    print(f"Claude: {assistant_message}")
    print()
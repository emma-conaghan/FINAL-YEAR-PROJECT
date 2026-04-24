import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
client = Anthropic(api_key=api_key)

conversation_history = []

def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
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
    
    return assistant_message

def main():
    print("Chat with Claude (type 'quit' to exit, 'clear' to clear history)")
    print("-" * 50)
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_history.clear()
            print("Conversation history cleared.")
            continue
        
        response = chat(user_input)
        print(f"Claude: {response}")
        print()

if __name__ == "__main__":
    main()
from anthropic import Anthropic

import os

api_key = os.environ.get("ANTHROPIC_API_KEY", "")

client = Anthropic(api_key=api_key)

conversation_history = []

system_prompt = "You are a helpful assistant. Be concise and clear in your responses."


def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8096,
        system=system_prompt,
        messages=conversation_history
    )

    assistant_message = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message


def main():
    print("Chat with Claude (type 'quit' to exit, 'reset' to clear history)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            conversation_history.clear()
            print("Conversation history cleared.")
            continue

        try:
            response = chat(user_input)
            print(f"\nClaude: {response}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
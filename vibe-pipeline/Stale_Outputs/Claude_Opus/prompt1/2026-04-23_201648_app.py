from anthropic import Anthropic

import os

api_key = os.environ.get("ANTHROPIC_API_KEY", "")

client = Anthropic(api_key=api_key)

conversation_history = []

def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
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
    print("Chat with Claude (type 'quit' to exit)")
    print("-" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = chat(user_input)
        print(f"\nClaude: {response}")


if __name__ == "__main__":
    main()
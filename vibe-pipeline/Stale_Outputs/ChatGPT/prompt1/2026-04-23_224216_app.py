from anthropic import Anthropic

api_key = "your_api_key_here"  # Replace with your actual API key
client = Anthropic(api_key=api_key)

def ask_anthropic(prompt):
    response = client.completions.create(
        model="claude-v1",
        prompt=prompt,
        max_tokens_to_sample=100,
        stop_sequences=["\n\n"],
    )
    return response.completion

if __name__ == "__main__":
    question = "What is the capital of France?"
    answer = ask_anthropic(question)
    print("Answer:", answer)
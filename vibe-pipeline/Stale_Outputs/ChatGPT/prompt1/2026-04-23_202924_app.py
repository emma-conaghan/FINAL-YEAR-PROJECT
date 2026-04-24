from anthropic import Anthropic

api_key = "your_api_key_here"
client = Anthropic(api_key=api_key)

def get_completion(prompt, model="claude-v1", max_tokens=300):
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens_to_sample=max_tokens,
        stop_sequences=["\n\nHuman:"]
    )
    return response.completion

if __name__ == "__main__":
    prompt = "Hello, how are you today?"
    completion = get_completion(prompt)
    print(completion)
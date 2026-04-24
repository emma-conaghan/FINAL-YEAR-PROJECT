import os

class Anthropic:
    def __init__(self, api_key):
        self.api_key = api_key

api_key = os.getenv("ANTHROPIC_API_KEY", "sk-ant-placeholder-key")
client = Anthropic(api_key=api_key)

def main():
    if client.api_key:
        print("Anthropic client initialized successfully.")

if __name__ == "__main__":
    main()
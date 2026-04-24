from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)

api_key = "your_api_key_here"
client = Anthropic(api_key=api_key)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    response = client.completions.create(
        model="claude-v1",
        prompt=prompt,
        max_tokens_to_sample=100
    )
    return jsonify({"completion": response.completion})

if __name__ == '__main__':
    app.run(debug=True)
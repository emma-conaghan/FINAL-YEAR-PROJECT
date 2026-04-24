from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)

api_key = "your_api_key_here"
client = Anthropic(api_key=api_key)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = client.completions.create(
        model="claude-v1",
        prompt=prompt,
        max_tokens_to_sample=300,
        stop_sequences=["\n\n"],
    )
    return jsonify({"response": response.completion})

if __name__ == "__main__":
    app.run(debug=True)
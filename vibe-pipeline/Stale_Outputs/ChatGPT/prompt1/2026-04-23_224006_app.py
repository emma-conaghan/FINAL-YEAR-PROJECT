from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)

api_key = "your_anthropic_api_key_here"
client = Anthropic(api_key=api_key)

@app.route("/complete", methods=["POST"])
def complete():
    data = request.json
    prompt = data.get("prompt", "")
    response = client.completions.create(
        model="claude-v1",
        prompt=prompt,
        max_tokens_to_sample=300,
        stop_sequences=["\nHuman:"]
    )
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
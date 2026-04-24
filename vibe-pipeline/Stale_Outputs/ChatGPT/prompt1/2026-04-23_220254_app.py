from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)

api_key = "your_api_key_here"
client = Anthropic(api_key=api_key)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    response = client.completions.create(
        model="claude-v1",
        prompt=prompt,
        max_tokens_to_sample=300,
        stop_sequences=["\n\nHuman:"]
    )
    return jsonify({"completion": response.completion})

if __name__ == '__main__':
    app.run(debug=True)
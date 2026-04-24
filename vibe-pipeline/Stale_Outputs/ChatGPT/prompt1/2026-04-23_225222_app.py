from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)
api_key = "your_anthropic_api_key"
client = Anthropic(api_key=api_key)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    try:
        response = client.completions.create(
            model="claude-v1",
            prompt=prompt,
            max_tokens_to_sample=100
        )
        return jsonify({"completion": response.completion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
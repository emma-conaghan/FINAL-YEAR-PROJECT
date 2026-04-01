import os
from anthropic import Anthropic


def generate_code(prompt, model_name="claude-sonnet-4-5"):
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is missing")

    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except Exception as e:
        raise SystemExit(f"Claude request failed: {e}")

    code_parts = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            code_parts.append(block.text)

    code = "".join(code_parts).strip()
    code = code.replace("```python", "").replace("```", "").strip()

    if len(code) < 20:
        raise SystemExit("Claude returned empty or too-short output")

    return code
import os
from anthropic import Anthropic


def generate_code(prompt, model_name="claude-sonnet-4-5"):
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is missing")

    client = Anthropic(api_key=api_key)

    full_prompt = (
    "Return ONLY valid Python 3 code for a single file called app.py. "
    "The code may be insecure or poor quality if requested, but it MUST remain syntactically valid Python 3. "
    "Do not use Python 2 syntax. "
    "Do not use duplicate parameter names. "
    "Do not include markdown, backticks, or explanations.\n\n"
    + prompt
)

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )
    except Exception as e:
        raise RuntimeError(f"Claude request failed: {e}")

    code_parts = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            code_parts.append(block.text)

    code = "".join(code_parts).strip()
    code = code.replace("```python", "").replace("```", "").strip()

    if len(code) < 20:
        raise Exception("Claude returned empty or too-short output")

    return code
from openai import OpenAI
import os


def generate_code(prompt, model_name="gpt-4.1-mini"):
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)

    full_prompt = (
    "Return ONLY valid Python 3 code for a single file called app.py. "
    "The code may be insecure or poor quality if requested, but it MUST remain syntactically valid Python 3. "
    "Do not use Python 2 syntax. "
    "Do not use duplicate parameter names. "
    "Do not include markdown, backticks, or explanations.\n\n"
    + prompt
)


    try:
        response = client.responses.create(
            model=model_name,
            input=full_prompt
        )
    except Exception as e:
        raise RuntimeError(f"ChatGPT request failed: {e}")

    code = (response.output_text or "").strip()
    code = code.replace("```python", "").replace("```", "").strip()

    if len(code) < 20:
        raise Exception("ChatGPT returned empty or too-short output")

    return code
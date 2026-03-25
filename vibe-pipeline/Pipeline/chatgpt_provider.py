from openai import OpenAI
import os


def generate_code(prompt, model_name="gpt-4.1-mini"):
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)

    full_prompt = (
        "Return ONLY valid Python code for a single file called app.py. "
        "No markdown, no backticks, no explanation.\n\n"
        + prompt
    )

    try:
        response = client.responses.create(
            model=model_name,
            input=full_prompt
        )
    except Exception as e:
        raise SystemExit(f"ChatGPT request failed: {e}")

    code = (response.output_text or "").strip()
    code = code.replace("```python", "").replace("```", "").strip()

    if len(code) < 20:
        raise SystemExit("ChatGPT returned empty or too-short output")

    return code
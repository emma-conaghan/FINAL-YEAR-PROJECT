import os
import requests


def generate_code(prompt, model_name="openai/gpt-4.1"):
    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        raise SystemExit("GITHUB_TOKEN is missing")

    full_prompt = (
    "Return ONLY valid Python 3 code for a single file called app.py. "
    "The code may be insecure or poor quality if requested, but it MUST remain syntactically valid Python 3. "
    "Do not use Python 2 syntax. "
    "Do not use duplicate parameter names. "
    "Do not include markdown, backticks, or explanations.\n\n"
    + prompt
)

    url = "https://models.github.ai/inference/chat/completions"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": full_prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"GitHub Models request failed: {e}")

    data = response.json()
    code = data["choices"][0]["message"]["content"].strip()
    code = code.replace("```python", "").replace("```", "").strip()

    if len(code) < 20:
        raise Exception("GitHub Models returned empty or too-short output")

    return code
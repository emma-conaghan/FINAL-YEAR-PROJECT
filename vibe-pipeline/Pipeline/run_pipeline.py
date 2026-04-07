import os
import re
import datetime
from pathlib import Path

from google import genai
from dotenv import load_dotenv

from config import (
    PROMPT_FILE,
    SCORESHEET,
    PROJECT_KEY,
    MODEL_NAME,
    GEMINI_OUTPUT_DIR,
    CHATGPT_OUTPUT_DIR,
    CLAUDE_OUTPUT_DIR,
    CHATGPT_MODEL,
    CLAUDE_MODEL,
)
from sonar_runner import run_sonar
from sonar_issues import export_sonar_issues
from chatgpt_provider import generate_code as generate_chatgpt_code
from claude_provider import generate_code as generate_claude_code

import shutil

def move_to_stale(file_path):
    stale_path = file_path.replace("Outputs", "Stale_Outputs")

    os.makedirs(os.path.dirname(stale_path), exist_ok=True)

    shutil.move(file_path, stale_path)

    print(f"Moved to stale: {stale_path}")

def check_for_secrets():
    repo_root = Path(__file__).resolve().parents[1]

    patterns = [
        r"AIza[0-9A-Za-z\-_]{35}",
        r"sqp_[0-9a-f]{40}",
        r"sk-[A-Za-z0-9]{20,}",
    ]

    allowed_files = {".py", ".txt", ".csv", ".properties"}
    excluded_dirs = {"venv", ".venv", ".scannerwork", "Outputs", "Stale_Outputs", "Generated_code"}

    for file in repo_root.rglob("*"):
        if not file.is_file():
            continue

        if file.suffix not in allowed_files:
            continue

        if any(part in excluded_dirs for part in file.parts):
            continue

        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                if re.search(pattern, content):
                    raise SystemExit(
                        f"Potential secret detected in {file}. "
                        "Remove it from the file and store it in .env instead."
                    )
        except UnicodeDecodeError:
            continue

load_dotenv(override=True)
check_for_secrets()


def save_output(code, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    timestamp_file = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archived_file = os.path.join(output_dir, f"{timestamp_file}_app.py")

    with open(archived_file, "w", encoding="utf-8") as f:
        f.write(code)

    return archived_file


def append_scoresheet(model_name, archived_file):
    os.makedirs("Scoring", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_file = (not os.path.exists(SCORESHEET)) or (os.path.getsize(SCORESHEET) == 0)

    with open(SCORESHEET, "a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp,model,output_file,notes\n")
        f.write(f"{timestamp},{model_name},{archived_file},pipeline ran ok\n")


def main():
    # 1) Read prompt
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt = f.read()

    full_prompt = (
        "Return ONLY valid Python code for a single file called app.py. "
        "No markdown, no backticks, no explanation.\n\n"
        + prompt
    )

    # 2) Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    print("GEMINI_API_KEY set?", "YES" if api_key else "NO")

    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=api_key)

    print("Calling Gemini...")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt
        )
    except Exception as e:
        raise SystemExit(f"Gemini request failed: {e}")

    gemini_code = (response.text or "").strip()
    gemini_code = gemini_code.replace("```python", "").replace("```", "").strip()

    if len(gemini_code) < 20:
        raise SystemExit("Gemini returned empty or too-short output.")

    gemini_archived_file = save_output(gemini_code, GEMINI_OUTPUT_DIR)
    print("Saved Gemini output to:", gemini_archived_file)
    append_scoresheet("gemini", gemini_archived_file)

    # 3) ChatGPT
    print("Calling ChatGPT...")
    chatgpt_code = generate_chatgpt_code(full_prompt, CHATGPT_MODEL)

    chatgpt_archived_file = save_output(chatgpt_code, CHATGPT_OUTPUT_DIR)
    print("Saved ChatGPT output to:", chatgpt_archived_file)
    append_scoresheet("chatgpt", chatgpt_archived_file)

    # 3b) Claude
    try:
        print("Calling Claude...")
        claude_code = generate_claude_code(full_prompt, CLAUDE_MODEL)
        claude_archived_file = save_output(claude_code, CLAUDE_OUTPUT_DIR)
        print("Saved Claude output to:", claude_archived_file)
        append_scoresheet("claude", claude_archived_file)
    except Exception as e:
        print(f"Claude failed: {e}")
        claude_archived_file = None

    # 4) Run Sonar ONCE on all Outputs
    print("Running Sonar scan...")
    run_sonar()
    print("Sonar scan completed")

    # 5) Export Sonar issues ONCE
    sonar_login = os.environ.get("SONAR_LOGIN")

    export_sonar_issues(
        project_key=PROJECT_KEY,
        sonar_login=sonar_login,
        csv_file="Scoring/sonar_issues.csv"
    )

    print("Exported Sonar issues to: Scoring/sonar_issues.csv")

    # 6) Move analysed files to stale storage
    move_to_stale(gemini_archived_file)
    move_to_stale(chatgpt_archived_file)
    if claude_archived_file:
        move_to_stale(claude_archived_file)

    print("Updated:", SCORESHEET)
    print("DONE")


if __name__ == "__main__":
    main()
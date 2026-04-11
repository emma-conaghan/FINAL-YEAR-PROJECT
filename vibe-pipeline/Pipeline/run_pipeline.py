import os
import re
import datetime
import ast
from pathlib import Path
import shutil
import sys
import subprocess

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
    GITHUB_MODELS_OUTPUT_DIR,
    GITHUB_MODELS_MODEL,
    CLAUDE_OPUS_MODEL,
    CLAUDE_OPUS_OUTPUT_DIR,
)
from sonar_runner import run_sonar
from sonar_issues import export_sonar_issues
from chatgpt_provider import generate_code as generate_chatgpt_code
from claude_provider import generate_code as generate_claude_code
from github_models_provider import generate_code as generate_github_models_code


def move_to_stale(file_path):
    stale_path = file_path.replace("Outputs", "Stale_Outputs")

    os.makedirs(os.path.dirname(stale_path), exist_ok=True)
    shutil.move(file_path, stale_path)

    print(f"Moved to stale: {stale_path}")


def move_invalid_output(file_path):
    invalid_path = file_path.replace("Outputs", "Invalid_Outputs")

    os.makedirs(os.path.dirname(invalid_path), exist_ok=True)
    shutil.move(file_path, invalid_path)

    print(f"Moved invalid file to: {invalid_path}")
    return invalid_path


def check_for_secrets():
    repo_root = Path(__file__).resolve().parents[1]

    patterns = [
        r"AIza[0-9A-Za-z\-_]{35}",
        r"sqp_[0-9a-f]{40}",
        r"sk-[A-Za-z0-9]{20,}",
    ]

    allowed_files = {".py", ".txt", ".csv", ".properties"}
    excluded_dirs = {"venv", ".venv", ".scannerwork", "Outputs", "Stale_Outputs", "Invalid_Outputs", "Generated_code"}

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


def check_python_syntax(code):
    """
    Safely checks whether generated code is valid Python syntax.
    This does NOT execute the code.
    Returns (True, "") if valid, otherwise (False, error_message).
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def save_output(code, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    timestamp_file = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archived_file = os.path.join(output_dir, f"{timestamp_file}_app.py")

    with open(archived_file, "w", encoding="utf-8") as f:
        f.write(code)

    return archived_file


def append_scoresheet(model_name, archived_file, status, syntax_valid, notes):
    os.makedirs("Scoring", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_file = (not os.path.exists(SCORESHEET)) or (os.path.getsize(SCORESHEET) == 0)

    # Clean commas/newlines so the CSV doesn't get mangled
    safe_notes = str(notes).replace("\n", " ").replace(",", ";")

    with open(SCORESHEET, "a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp,model,output_file,status,syntax_valid,notes\n")
        f.write(f"{timestamp},{model_name},{archived_file},{status},{syntax_valid},{safe_notes}\n")


def main():
    # 1) Read prompt
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt = f.read()

    full_prompt = (
    "Return ONLY valid Python 3 code for a single file called app.py. "
    "The code may be insecure or poor quality if requested, but it MUST remain syntactically valid Python 3. "
    "Do not use Python 2 syntax. "
    "Do not use duplicate parameter names. "
    "Do not include markdown, backticks, or explanations.\n\n"
    + prompt
)

    # Default flags so they always exist later
    gemini_valid = False
    chatgpt_valid = False
    claude_valid = False
    github_valid = False
    claude_opus_valid = False

    gemini_archived_file = None
    chatgpt_archived_file = None
    claude_archived_file = None
    github_models_archived_file = None
    claude_opus_archived_file = None   

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

    gemini_valid, gemini_error = check_python_syntax(gemini_code)
    print("Gemini syntax valid?", gemini_valid)

    if not gemini_valid:
        print("Gemini syntax error:", gemini_error)
        gemini_archived_file = move_invalid_output(gemini_archived_file)

    append_scoresheet(
        "gemini",
        gemini_archived_file,
        "valid_python" if gemini_valid else "invalid_python",
        gemini_valid,
        gemini_error if gemini_error else "pipeline ran ok"
    )

    # 3) ChatGPT
    print("Calling ChatGPT...")
    chatgpt_code = generate_chatgpt_code(full_prompt, CHATGPT_MODEL)

    chatgpt_archived_file = save_output(chatgpt_code, CHATGPT_OUTPUT_DIR)
    print("Saved ChatGPT output to:", chatgpt_archived_file)

    chatgpt_valid, chatgpt_error = check_python_syntax(chatgpt_code)
    print("ChatGPT syntax valid?", chatgpt_valid)

    if not chatgpt_valid:
        print("ChatGPT syntax error:", chatgpt_error)
        chatgpt_archived_file = move_invalid_output(chatgpt_archived_file)

    append_scoresheet(
        "chatgpt",
        chatgpt_archived_file,
        "valid_python" if chatgpt_valid else "invalid_python",
        chatgpt_valid,
        chatgpt_error if chatgpt_error else "pipeline ran ok"
    )

    # 3a) Claude
    try:
        print("Calling Claude...")
        claude_code = generate_claude_code(full_prompt, CLAUDE_MODEL)

        claude_archived_file = save_output(claude_code, CLAUDE_OUTPUT_DIR)
        print("Saved Claude output to:", claude_archived_file)

        claude_valid, claude_error = check_python_syntax(claude_code)
        print("Claude syntax valid?", claude_valid)

        if not claude_valid:
            print("Claude syntax error:", claude_error)
            claude_archived_file = move_invalid_output(claude_archived_file)

        append_scoresheet(
            "claude",
            claude_archived_file,
            "valid_python" if claude_valid else "invalid_python",
            claude_valid,
            claude_error if claude_error else "pipeline ran ok"
        )

    except Exception as e:
        print(f"Claude failed: {e}")
        claude_archived_file = None
        append_scoresheet("claude", "N/A", "generation_failed", False, str(e))

    # 3b) Claude Opus
    try:
        print("Calling Claude Opus...")
        claude_opus_code = generate_claude_code(full_prompt, CLAUDE_OPUS_MODEL)

        claude_opus_archived_file = save_output(claude_opus_code, CLAUDE_OPUS_OUTPUT_DIR)
        print("Saved Claude Opus output to:", claude_opus_archived_file)

        claude_opus_valid, claude_opus_error = check_python_syntax(claude_opus_code)
        print("Claude Opus syntax valid?", claude_opus_valid)
        if not claude_opus_valid:
            print("Claude Opus syntax error:", claude_opus_error)
            claude_opus_archived_file = move_invalid_output(claude_opus_archived_file)

        append_scoresheet(
            "claude_opus",
            claude_opus_archived_file,
            "valid_python" if claude_opus_valid else "invalid_python",
            claude_opus_valid,
            claude_opus_error if claude_opus_error else "pipeline ran ok"
        )

    except Exception as e:
        print(f"Claude Opus failed: {e}")
        claude_opus_archived_file = None
        append_scoresheet("claude_opus", "N/A", "generation_failed", False, str(e))


    # 4) GitHub Models / Copilot-style
    print("Calling GitHub Models...")
    github_models_code = generate_github_models_code(prompt, GITHUB_MODELS_MODEL)

    github_models_archived_file = save_output(github_models_code, GITHUB_MODELS_OUTPUT_DIR)
    print("Saved GitHub Models output to:", github_models_archived_file)

    github_valid, github_error = check_python_syntax(github_models_code)
    print("GitHub Models syntax valid?", github_valid)

    if not github_valid:
        print("GitHub Models syntax error:", github_error)
        github_models_archived_file = move_invalid_output(github_models_archived_file)

    append_scoresheet(
        "copilot",
        github_models_archived_file,
        "valid_python" if github_valid else "invalid_python",
        github_valid,
        github_error if github_error else "pipeline ran ok"
    )

    # 5) Run Sonar ONCE on valid files left in Outputs
    print("Running Sonar scan...")
    run_sonar()
    print("Sonar scan completed")

    # 6) Export Sonar issues ONCE
    sonar_login = os.environ.get("SONAR_LOGIN")

    export_sonar_issues(
        project_key=PROJECT_KEY,
        sonar_login=sonar_login,
        csv_file="Scoring/sonar_issues.csv"
    )

    subprocess.run([sys.executable, "Pipeline/analysis.py"], check=True)

    # 7) Move analysed VALID files to stale storage
    if gemini_valid and gemini_archived_file:
        move_to_stale(gemini_archived_file)

    if chatgpt_valid and chatgpt_archived_file:
        move_to_stale(chatgpt_archived_file)

    if claude_valid and claude_archived_file:
        move_to_stale(claude_archived_file)

    if github_valid and github_models_archived_file:
        move_to_stale(github_models_archived_file)

    if claude_opus_valid and claude_opus_archived_file:
        move_to_stale(claude_opus_archived_file)



    print("Updated:", SCORESHEET)
    print("DONE")


if __name__ == "__main__":
    main()
import os
import re
import ast
import sys
import shutil
import datetime
import subprocess
from pathlib import Path

from google import genai
from dotenv import load_dotenv

from config import (
    PROMPTS_DIR,
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


# Prompt loading helpers

def get_prompt_files(prompts_dir):
    """
    Return all .txt prompt files from the Prompts directory in sorted order.
    """
    if not os.path.exists(prompts_dir):
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompt_files = [
        f for f in os.listdir(prompts_dir)
        if f.lower().endswith(".txt")
    ]

    if not prompt_files:
        raise ValueError(f"No .txt prompt files found in {prompts_dir}")

    return sorted(prompt_files)


def make_prompt_id(filename):
    """
    Convert a filename like 'prompt_1.txt' into a clean prompt_id like 'prompt_1'.
    """
    prompt_id = os.path.splitext(filename)[0]
    prompt_id = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt_id)
    return prompt_id


def load_prompts(prompts_dir):
    """
    Load all prompt files and return them as a list of dictionaries.

    Example:
    [
        {
            "prompt_id": "prompt_1",
            "filename": "prompt_1.txt",
            "prompt_text": "Write a Python app..."
        }
    ]
    """
    prompts = []

    for filename in get_prompt_files(prompts_dir):
        full_path = os.path.join(prompts_dir, filename)

        with open(full_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()

        prompts.append({
            "prompt_id": make_prompt_id(filename),
            "filename": filename,
            "prompt_text": prompt_text
        })

    return prompts


# File movement helpers

def move_to_stale(file_path):
    """
    Move a valid analysed output file from Outputs to Stale_Outputs.
    """
    stale_path = file_path.replace("Outputs", "Stale_Outputs")

    os.makedirs(os.path.dirname(stale_path), exist_ok=True)
    shutil.move(file_path, stale_path)

    print(f"Moved to stale: {stale_path}")


def move_invalid_output(file_path):
    """
    Move an invalid Python file from Outputs to Invalid_Outputs.
    """
    invalid_path = file_path.replace("Outputs", "Invalid_Outputs")

    os.makedirs(os.path.dirname(invalid_path), exist_ok=True)
    shutil.move(file_path, invalid_path)

    print(f"Moved invalid file to: {invalid_path}")
    return invalid_path


# Secret scanning

def check_for_secrets():
    """
    Scan the repository for obvious secrets before running the pipeline.
    This helps avoid accidentally committing or processing live tokens.
    """
    repo_root = Path(__file__).resolve().parents[1]

    patterns = [
        r"AIza[0-9A-Za-z\-_]{35}",
        r"sqp_[0-9a-f]{40}",
        r"sk-[A-Za-z0-9]{20,}",
    ]

    allowed_files = {".py", ".txt", ".csv", ".properties"}
    excluded_dirs = {
        "venv",
        ".venv",
        ".scannerwork",
        "Outputs",
        "Stale_Outputs",
        "Invalid_Outputs",
        "Generated_code"
    }

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


# Load environment variables and run secret scan once
load_dotenv(override=True)
check_for_secrets()

# Syntax checking

def check_python_syntax(code):
    """
    Check whether generated code is valid Python syntax.
    This does NOT execute the code.

    Returns:
        (True, "") if valid
        (False, error_message) if invalid
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


# Output and scoring helpers
def save_output(code, output_dir, prompt_id):
    """
    Save generated code into a prompt-specific folder.

    Example:
    Outputs/Gemini/prompt_1/2026-04-23_153000_app.py
    """
    prompt_output_dir = os.path.join(output_dir, prompt_id)
    os.makedirs(prompt_output_dir, exist_ok=True)

    timestamp_file = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archived_file = os.path.join(prompt_output_dir, f"{timestamp_file}_app.py")

    with open(archived_file, "w", encoding="utf-8") as f:
        f.write(code)

    return archived_file


def append_scoresheet(prompt_id, prompt_file, model_name, archived_file, status, syntax_valid, notes):
    """
    Append one pipeline result row to the scoresheet CSV.
    """
    os.makedirs("Scoring", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_file = (not os.path.exists(SCORESHEET)) or (os.path.getsize(SCORESHEET) == 0)

    # Keep CSV rows clean
    safe_notes = str(notes).replace("\n", " ").replace(",", ";")

    with open(SCORESHEET, "a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp,prompt_id,prompt_file,model,output_file,status,syntax_valid,notes\n")

        f.write(
            f"{timestamp},{prompt_id},{prompt_file},{model_name},{archived_file},{status},{syntax_valid},{safe_notes}\n"
        )


def build_full_prompt(prompt_text):
    """
    Wrap the user task prompt with rules that force the models to return
    syntactically valid Python for a single file only.
    """
    return (
        "Return ONLY valid Python 3 code for a single file called app.py. "
        "The code may be insecure or poor quality if requested, but it MUST remain syntactically valid Python 3. "
        "Do not use Python 2 syntax. "
        "Do not use duplicate parameter names. "
        "Do not include markdown, backticks, or explanations.\n\n"
        + prompt_text
    )


# Main pipeline

def main():
    """
    Main pipeline flow:

    1. Load all prompts from Prompts/
    2. For each prompt:
       - call each model
       - save output into prompt-specific folder
       - syntax check output
       - move invalid output if needed
       - log result to scoresheet
    3. Run Sonar once across all remaining valid files in Outputs/
    4. Export Sonar issues
    5. Run analysis script
    6. Move valid analysed files to Stale_Outputs
    """
    prompts = load_prompts(PROMPTS_DIR)

    print(f"Loaded {len(prompts)} prompt(s) from {PROMPTS_DIR}")

    # Gemini client setup
    api_key = os.environ.get("GEMINI_API_KEY")
    print("GEMINI_API_KEY set?", "YES" if api_key else "NO")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=api_key)

    # Keep track of valid files so we can move them to stale AFTER Sonar runs
    valid_files_to_archive = []

    for prompt_data in prompts:
        prompt_id = prompt_data["prompt_id"]
        prompt_file = prompt_data["filename"]
        prompt_text = prompt_data["prompt_text"]

        print(f"\n--- Running prompt: {prompt_id} ({prompt_file}) ---")

        full_prompt = build_full_prompt(prompt_text)


        # 1) Gemini
        try:
            print("Calling Gemini...")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=full_prompt
            )

            gemini_code = (response.text or "").strip()
            gemini_code = gemini_code.replace("```python", "").replace("```", "").strip()

            if len(gemini_code) < 20:
                raise ValueError("Gemini returned empty or too-short output.")

            gemini_archived_file = save_output(gemini_code, GEMINI_OUTPUT_DIR, prompt_id)
            print("Saved Gemini output to:", gemini_archived_file)

            gemini_valid, gemini_error = check_python_syntax(gemini_code)
            print("Gemini syntax valid?", gemini_valid)

            if not gemini_valid:
                print("Gemini syntax error:", gemini_error)
                gemini_archived_file = move_invalid_output(gemini_archived_file)

            append_scoresheet(
                prompt_id,
                prompt_file,
                "gemini",
                gemini_archived_file,
                "valid_python" if gemini_valid else "invalid_python",
                gemini_valid,
                gemini_error if gemini_error else "pipeline ran ok"
            )

            if gemini_valid:
                valid_files_to_archive.append(gemini_archived_file)

        except Exception as e:
            print(f"Gemini failed: {e}")
            append_scoresheet(
                prompt_id,
                prompt_file,
                "gemini",
                "N/A",
                "generation_failed",
                False,
                str(e)
            )
            pass

        # 2) ChatGPT
        try:
            print("Calling ChatGPT...")

            chatgpt_code = generate_chatgpt_code(full_prompt, CHATGPT_MODEL)

            chatgpt_archived_file = save_output(chatgpt_code, CHATGPT_OUTPUT_DIR, prompt_id)
            print("Saved ChatGPT output to:", chatgpt_archived_file)

            chatgpt_valid, chatgpt_error = check_python_syntax(chatgpt_code)
            print("ChatGPT syntax valid?", chatgpt_valid)

            if not chatgpt_valid:
                print("ChatGPT syntax error:", chatgpt_error)
                chatgpt_archived_file = move_invalid_output(chatgpt_archived_file)

            append_scoresheet(
                prompt_id,
                prompt_file,
                "chatgpt",
                chatgpt_archived_file,
                "valid_python" if chatgpt_valid else "invalid_python",
                chatgpt_valid,
                chatgpt_error if chatgpt_error else "pipeline ran ok"
            )

            if chatgpt_valid:
                valid_files_to_archive.append(chatgpt_archived_file)

        except Exception as e:
            print(f"ChatGPT failed: {e}")
            append_scoresheet(
                prompt_id,
                prompt_file,
                "chatgpt",
                "N/A",
                "generation_failed",
                False,
                str(e)
            )
            pass

        # 3) Claude
        try:
            print("Calling Claude...")

            claude_code = generate_claude_code(full_prompt, CLAUDE_MODEL)

            claude_archived_file = save_output(claude_code, CLAUDE_OUTPUT_DIR, prompt_id)
            print("Saved Claude output to:", claude_archived_file)

            claude_valid, claude_error = check_python_syntax(claude_code)
            print("Claude syntax valid?", claude_valid)

            if not claude_valid:
                print("Claude syntax error:", claude_error)
                claude_archived_file = move_invalid_output(claude_archived_file)

            append_scoresheet(
                prompt_id,
                prompt_file,
                "claude",
                claude_archived_file,
                "valid_python" if claude_valid else "invalid_python",
                claude_valid,
                claude_error if claude_error else "pipeline ran ok"
            )

            if claude_valid:
                valid_files_to_archive.append(claude_archived_file)

        except Exception as e:
            print(f"Claude failed: {e}")
            append_scoresheet(
                prompt_id,
                prompt_file,
                "claude",
                "N/A",
                "generation_failed",
                False,
                str(e)
            )
            pass

        # 4) Claude Opus
        try:
            print("Calling Claude Opus...")

            claude_opus_code = generate_claude_code(full_prompt, CLAUDE_OPUS_MODEL)

            claude_opus_archived_file = save_output(claude_opus_code, CLAUDE_OPUS_OUTPUT_DIR, prompt_id)
            print("Saved Claude Opus output to:", claude_opus_archived_file)

            claude_opus_valid, claude_opus_error = check_python_syntax(claude_opus_code)
            print("Claude Opus syntax valid?", claude_opus_valid)

            if not claude_opus_valid:
                print("Claude Opus syntax error:", claude_opus_error)
                claude_opus_archived_file = move_invalid_output(claude_opus_archived_file)

            append_scoresheet(
                prompt_id,
                prompt_file,
                "claude_opus",
                claude_opus_archived_file,
                "valid_python" if claude_opus_valid else "invalid_python",
                claude_opus_valid,
                claude_opus_error if claude_opus_error else "pipeline ran ok"
            )

            if claude_opus_valid:
                valid_files_to_archive.append(claude_opus_archived_file)

        except Exception as e:
            print(f"Claude Opus failed: {e}")
            append_scoresheet(
                prompt_id,
                prompt_file,
                "claude_opus",
                "N/A",
                "generation_failed",
                False,
                str(e)
            )
            pass

        # 5) GitHub Models
        try:
            print("Calling GitHub Models...")

            github_models_code = generate_github_models_code(full_prompt, GITHUB_MODELS_MODEL)

            github_models_archived_file = save_output(github_models_code, GITHUB_MODELS_OUTPUT_DIR, prompt_id)
            print("Saved GitHub Models output to:", github_models_archived_file)

            github_valid, github_error = check_python_syntax(github_models_code)
            print("GitHub Models syntax valid?", github_valid)

            if not github_valid:
                print("GitHub Models syntax error:", github_error)
                github_models_archived_file = move_invalid_output(github_models_archived_file)

            append_scoresheet(
                prompt_id,
                prompt_file,
                "copilot",
                github_models_archived_file,
                "valid_python" if github_valid else "invalid_python",
                github_valid,
                github_error if github_error else "pipeline ran ok"
            )

            if github_valid:
                valid_files_to_archive.append(github_models_archived_file)

        except BaseException as e:
            print(f"GitHub Models failed: {e}")
            append_scoresheet(
                prompt_id,
                prompt_file,
                "copilot",
                "N/A",
                "generation_failed",
                False,
                str(e)
            )

            pass


    # Run Sonar once after all prompt/model generations are complete
    print("\nRunning Sonar scan...")
    run_sonar()
    print("Sonar scan completed")

    sonar_login = os.environ.get("SONAR_LOGIN")

    export_sonar_issues(
        project_key=PROJECT_KEY,
        sonar_login=sonar_login,
        csv_file="Scoring/sonar_issues.csv"
    )

    # Run your analysis script after Sonar issue export
    subprocess.run([sys.executable, "Pipeline/analysis.py"], check=True)

    # Move valid analysed files to stale storage
    for file_path in valid_files_to_archive:
        if os.path.exists(file_path):
            move_to_stale(file_path)

    print("Updated:", SCORESHEET)
    print("DONE")


if __name__ == "__main__":
    main()
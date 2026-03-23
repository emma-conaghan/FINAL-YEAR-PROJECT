import os
import re
import datetime
import shutil
import subprocess
from pathlib import Path

from google import genai
from dotenv import load_dotenv
from sonar_issues import export_sonar_issues
from config import PROJECT_KEY


def check_for_secrets():
    """
    Stop the pipeline if a likely API key or token
    is found in repository files.
    """
    repo_root = Path(__file__).resolve().parents[1]

    patterns = [
        r"AIza[0-9A-Za-z\-_]{35}",   # Google API key
        r"sqp_[0-9a-f]{40}",         # Sonar token
        r"sk-[A-Za-z0-9]{20,}",      # OpenAI-style key
    ]

    allowed_files = {".py", ".txt", ".csv", ".properties"}

    for file in repo_root.rglob("*"):
        if not file.is_file():
            continue

        if file.suffix not in allowed_files:
            continue

        if ".venv" in file.parts or "venv" in file.parts or ".scannerwork" in file.parts:
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


load_dotenv()
check_for_secrets()

PROMPT_FILE = "Prompts/prompt.txt"
OUT_FILE = "Generated_code/app.py"
SCORESHEET = "Scoring/scoresheet.csv"
OUTPUT_DIR = "Outputs/Gemini"
MODEL_NAME = "gemini-3-flash-preview"


# 1) Read prompt
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt = f.read()


# 2) Call Gemini
api_key = os.environ.get("GEMINI_API_KEY")
print("GEMINI_API_KEY set?", "YES" if api_key else "NO")

if not api_key:
    raise SystemExit("GEMINI_API_KEY is missing. Put it in .env or export it in terminal.")

client = genai.Client(api_key=api_key)

full_prompt = (
    "Return ONLY valid Python code for a single file called app.py. "
    "No markdown, no backticks, no explanation.\n\n"
    + prompt
)

print("Calling Gemini...")

try:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt
    )
except Exception as e:
    raise SystemExit(f"Gemini request failed: {e}")

code = (response.text or "").strip()

# Remove markdown fences if they appear
code = code.replace("```python", "").replace("```", "").strip()

print("Gemini text length:", len(code))

if len(code) < 20:
    raise SystemExit("Gemini returned empty or too-short output.")


# 3) Save archived output
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp_file = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
archived_file = os.path.join(OUTPUT_DIR, f"{timestamp_file}_app.py")

with open(archived_file, "w", encoding="utf-8") as f:
    f.write(code)

print("Saved archived output to:", archived_file)


# 4) Copy to scan target
os.makedirs("Generated_code", exist_ok=True)
shutil.copyfile(archived_file, OUT_FILE)

print("Updated scan target:", OUT_FILE)

# 5) Run Sonar scan on the new archived file
print("Running Sonar scan...")

repo_root = Path(__file__).resolve().parents[1]
sonar_login = os.environ.get("SONAR_LOGIN")

if not sonar_login:
    raise SystemExit("SONAR_LOGIN environment variable not set.")

archived_path = Path(archived_file)
source_dir = str(archived_path.parent)
file_name = archived_path.name

result = subprocess.run(
    [
        "sonar-scanner",
        f"-Dsonar.login={sonar_login}",
        f"-Dsonar.projectKey={PROJECT_KEY}",
        "-Dsonar.host.url=http://localhost:9000",
        f"-Dsonar.sources={source_dir}",
        f"-Dsonar.inclusions={file_name}",
    ],
    cwd=str(repo_root)
)

if result.returncode != 0:
    raise SystemExit("Sonar scan failed")

print("Sonar scan completed")

sonar_login = os.environ.get("SONAR_LOGIN")

export_sonar_issues(
    project_key=PROJECT_KEY,
    sonar_login=sonar_login,
    model_name="gemini",
    output_file=archived_file,
    csv_file="Scoring/sonar_issues.csv"
)

print("Exported Sonar issues to: Scoring/sonar_issues.csv")

# 6) Update scoresheet
os.makedirs("Scoring", exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
new_file = (not os.path.exists(SCORESHEET)) or (os.path.getsize(SCORESHEET) == 0)

with open(SCORESHEET, "a", encoding="utf-8") as f:
    if new_file:
        f.write("timestamp,model,output_file,notes\n")
    f.write(f"{timestamp},gemini,{archived_file},pipeline ran ok\n")

print("Updated:", SCORESHEET)
print("DONE")
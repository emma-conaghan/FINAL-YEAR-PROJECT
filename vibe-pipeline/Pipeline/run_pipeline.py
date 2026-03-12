import os
import datetime
import shutil
import subprocess
from pathlib import Path

from google import genai
from dotenv import load_dotenv

load_dotenv()

PROMPT_FILE = "Prompts/prompt.txt"
OUT_FILE = "Generated_code/app.py"
SCORESHEET = "Scoring/scoresheet.csv"
OUTPUT_DIR = "Outputs/Gemini"


# 1) Read prompt
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt = f.read()



# 2) Call Gemini
api_key = os.environ.get("GEMINI_API_KEY")
print("GEMINI_API_KEY set?", "YES" if api_key else "NO")

if not api_key:
    raise SystemExit("GEMINI_API_KEY is missing. Run: export GEMINI_API_KEY='your_key'")

client = genai.Client(api_key=api_key)

full_prompt = (
    "Return ONLY valid Python code for a single file called app.py. "
    "No markdown, no backticks, no explanation.\n\n"
    + prompt
)

print("Calling Gemini...")
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=full_prompt
)

code = (response.text or "").strip()

# Remove markdown fences if the model adds them
code = code.replace("```python", "").replace("```", "").strip()

print("Gemini text length:", len(code))

if len(code) < 20:
    raise SystemExit("Gemini returned empty or too-short output.")


# 3) Save output to archive
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp_file = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
archived_file = f"{OUTPUT_DIR}/{timestamp_file}_app.py"

with open(archived_file, "w", encoding="utf-8") as f:
    f.write(code)

print("Saved archived output to:", archived_file)


# 4) Copy to scan target
os.makedirs("Generated_code", exist_ok=True)

shutil.copyfile(archived_file, OUT_FILE)

print("Updated scan target:", OUT_FILE)


# 5) Run Sonar scan
print("Running Sonar scan...")

repo_root = Path(__file__).resolve().parents[1]
sonar_token = os.environ.get("SONAR_LOGIN")

if not sonar_token:
    raise SystemExit("SONAR_LOGIN environment variable not set")

result = subprocess.run(
    ["sonar-scanner", f"-Dsonar.login={sonar_token}"],
    cwd=str(repo_root)
)

if result.returncode != 0:
    raise SystemExit("Sonar scan failed")

print("Sonar scan completed")


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
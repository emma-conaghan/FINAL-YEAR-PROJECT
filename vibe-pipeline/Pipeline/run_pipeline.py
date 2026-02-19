import os
import datetime
from google import genai

PROMPT_FILE = "Prompts/prompt.txt"
OUT_FILE = "Generated_code/app.py"
SCORESHEET = "Scoring/scoresheet.csv"

# 1) Read prompt
with open(PROMPT_FILE, "r") as f:
    prompt = f.read()

## 2) Call Gemini to generate Python code
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

# remove markdown fences if they appear
code = code.replace("```python", "").replace("```", "").strip()

print("Gemini text length:", len(code))

if len(code) < 20:
    raise SystemExit("Gemini returned empty/too-short output.")

# 3) Save generated code
os.makedirs("Generated_code", exist_ok=True)
with open(OUT_FILE, "w") as f:
    f.write(code)

print(" Saved generated code to:", OUT_FILE)

# 4) Run Sonar scan (uses sonar-project.properties)
print(" Running Sonar scan...")
exit_code = os.system("sonar-scanner")

if exit_code != 0:
    print("Sonar scan failed. Check terminal output above.")
    raise SystemExit(1)

print("Sonar scan completed")

# 5) Update scoresheet
os.makedirs("Scoring", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_file = (not os.path.exists(SCORESHEET)) or (os.path.getsize(SCORESHEET) == 0)

with open(SCORESHEET, "a") as f:
    if new_file:
        f.write("timestamp,notes\n")
    f.write(f"{timestamp},pipeline ran ok\n")

print("Updated:", SCORESHEET)
print(" DONE")

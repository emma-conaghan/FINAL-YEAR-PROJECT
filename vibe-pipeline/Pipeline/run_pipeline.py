import os
import datetime

PROMPT_FILE = "Prompts/prompt.txt"
OUT_FILE = "Generated_code/app.py"
SCORESHEET = "Scoring/scoresheet.csv"

# 1) Read prompt
with open(PROMPT_FILE, "r") as f:
    prompt = f.read()

# 2) Fake "LLM generated" code (placeholder)
code = f"""# Generated from prompt:
# {prompt}

print("Hello from generated code!")
"""

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

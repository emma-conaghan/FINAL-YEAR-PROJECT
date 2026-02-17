import os
import datetime

PROMPT_FILE = "Prompts/prompt.txt"
OUT_FILE = "Generated_code/app.py"
SCORESHEET = "Scoring/scoresheet.csv"

# read prompt
with open(PROMPT_FILE, "r") as f:
    prompt = f.read()

# fake generated code (placeholder)
code = f"""# Generated from prompt:
# {prompt}

print("Hello from generated code!")
"""

# save generated code
os.makedirs("Generated_code", exist_ok=True)
with open(OUT_FILE, "w") as f:
    f.write(code)

print("Saved generated code to:", OUT_FILE)

# run sonar scan
print("Running Sonar scan...")
os.system("sonar-scanner")

# update scoresheet
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_file = not os.path.exists(SCORESHEET) or os.path.getsize(SCORESHEET) == 0

with open(SCORESHEET, "a") as f:
    if new_file:
        f.write("timestamp,notes\n")
    f.write(f"{timestamp},sonar ran\n")

print("Updated:", SCORESHEET)
print("DONE")
    
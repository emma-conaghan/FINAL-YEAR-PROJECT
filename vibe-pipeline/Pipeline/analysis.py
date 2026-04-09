import os
import pandas as pd
import matplotlib.pyplot as plt

SCORESHEET = "Scoring/scoresheet.csv"
SONAR_ISSUES = "Scoring/sonar_issues.csv"
OUTPUT_DIR = "Scoring/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD DATA
scores_df = pd.read_csv(SCORESHEET)
sonar_df = pd.read_csv(SONAR_ISSUES)

print("Raw scoresheet columns:", scores_df.columns.tolist())
print("Raw sonar columns:", sonar_df.columns.tolist())

# Clean column names in case of whitespace issues
scores_df.columns = scores_df.columns.str.strip()
sonar_df.columns = sonar_df.columns.str.strip()

print("Cleaned scoresheet columns:", scores_df.columns.tolist())
print("Cleaned sonar columns:", sonar_df.columns.tolist())

# Make sure required scoresheet column exists
if "syntax_valid" not in scores_df.columns:
    raise ValueError(f"'syntax_valid' column not found. Columns found: {scores_df.columns.tolist()}")

# Convert syntax_valid values to real booleans
scores_df["syntax_valid"] = scores_df["syntax_valid"].astype(str).str.strip().str.lower().map({
    "true": True,
    "false": False
})

# Work out which Sonar column contains the file path
possible_path_columns = ["component", "file", "file_path", "path", "filename", "source"]

sonar_path_column = None
for col in possible_path_columns:
    if col in sonar_df.columns:
        sonar_path_column = col
        break

if sonar_path_column is None:
    raise ValueError(
        f"Could not find a file path column in sonar_issues.csv. Columns found: {sonar_df.columns.tolist()}"
    )

if "severity" not in sonar_df.columns:
    raise ValueError(
        f"Expected 'severity' column in sonar_issues.csv but it was not found. Columns found: {sonar_df.columns.tolist()}"
    )

print("Using sonar path column:", sonar_path_column)

# 1. SYNTAX VALIDITY ANALYSIS
syntax_summary = (
    scores_df.groupby("model")["syntax_valid"]
    .agg(["count", "sum"])
    .rename(columns={"count": "total", "sum": "valid"})
)

syntax_summary["valid_rate"] = syntax_summary["valid"] / syntax_summary["total"]

print("\n=== Syntax Validity ===")
print(syntax_summary)

syntax_summary.to_csv(f"{OUTPUT_DIR}/syntax_summary.csv", index_label="model")

# 2. ISSUE COUNT PER MODEL
def extract_model(file_path):
    file_path = str(file_path)

    if "Gemini" in file_path:
        return "gemini"
    if "ChatGPT" in file_path:
        return "chatgpt"
    if "Claude_Opus" in file_path:
        return "claude_opus"
    if "Claude" in file_path:
        return "claude"
    if "Copilot" in file_path:
        return "copilot"
    return "unknown"


sonar_df["model"] = sonar_df[sonar_path_column].astype(str).apply(extract_model)

issue_counts = sonar_df.groupby("model").size().rename("issue_count")

print("\n=== Issue Counts ===")
print(issue_counts)

issue_counts.to_csv(f"{OUTPUT_DIR}/issue_counts.csv", index_label="model")

# 3. SEVERITY BREAKDOWN
severity_counts = sonar_df.groupby(["model", "severity"]).size().unstack(fill_value=0)

print("\n=== Severity Breakdown ===")
print(severity_counts)

severity_counts.to_csv(f"{OUTPUT_DIR}/severity_breakdown.csv", index_label="model")

# 4. COMBINED METRICS
combined = syntax_summary.join(issue_counts, how="left").fillna(0)

combined["issue_count"] = combined["issue_count"].astype(int)
combined["issues_per_valid_file"] = combined["issue_count"] / combined["valid"].replace(0, 1)

print("\n=== Combined Metrics ===")
print(combined)

combined.to_csv(f"{OUTPUT_DIR}/combined_metrics.csv", index_label="model")

# 5. CHARTS
combined["valid_rate"].plot(kind="bar", title="Syntax Validity Rate")
plt.ylabel("Proportion of Valid Files")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/validity_rate.png")
plt.clf()

combined["issues_per_valid_file"].plot(kind="bar", title="Issues per Valid File")
plt.ylabel("Issues per Valid File")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/issues_per_file.png")
plt.clf()

print("\nAnalysis complete. Files saved to:", OUTPUT_DIR)
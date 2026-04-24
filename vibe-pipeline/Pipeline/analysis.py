import os
import pandas as pd
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

SONAR_LOGIN = os.environ.get("SONAR_LOGIN")
SONAR_URL = os.environ.get("SONAR_URL", "http://localhost:9000")
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

# RULE NAME LOOKUP
rule_name_cache = {}

def get_rule_name(rule_key):
    """
    Looks up a Sonar rule key like python:S1481
    and returns a human-readable rule name.
    Falls back to the raw key if lookup fails.
    """
    if pd.isna(rule_key):
        return "unknown_rule"

    rule_key = str(rule_key).strip()

    if rule_key in rule_name_cache:
        return rule_name_cache[rule_key]

    if not SONAR_LOGIN:
        print(f"No SONAR_LOGIN found. Using raw rule key for {rule_key}")
        rule_name_cache[rule_key] = rule_key
        return rule_key

    try:
        response = requests.get(
            f"{SONAR_URL}/api/rules/show",
            params={"key": rule_key},
            headers={"Authorization": f"Bearer {SONAR_LOGIN}"},
            timeout=30
        )

        if response.status_code != 200:
            print(f"Rule lookup failed for {rule_key}: {response.status_code} - {response.text}")
            rule_name_cache[rule_key] = rule_key
            return rule_key

        data = response.json()
        rule_name = data.get("rule", {}).get("name", rule_key)

        rule_name_cache[rule_key] = rule_name
        return rule_name

    except Exception as e:
        print(f"Exception looking up rule {rule_key}: {e}")
        rule_name_cache[rule_key] = rule_key
        return rule_key

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

MANUAL_RULE_MAP = {
    "python:S1481": "Unused local variables should be removed",
    "python:S1066": "Mergeable if statements should be combined",
    "python:S5797": "Exception handlers should preserve original exceptions",
    "python:S108": "Nested blocks of code should not be left empty",
    "python:S1172": "Unused function parameters should be removed",
    "python:S112": "Generic exceptions should never be raised",
    "python:S1186": "Functions and methods should not be empty",
    "python:S5603": "Regular expressions should not always fail",
    "python:S5806": "Regular expressions should not be too complicated",
    "python:S1764": "Identical expressions should not be used on both sides of a binary operator",
    "python:S3923": "All branches in a conditional structure should not have exactly the same implementation",
    "python:S5754": "Cryptographic hashes should not be vulnerable",
    "python:S5720": "Regular expressions should not be vulnerable to denial of service attacks",
    "python:S3776": "Cognitive Complexity of functions should not be too high",
    "python:S2757": "Assertions should not test for truthiness of tuples",
    "python:S1763": "Code should not be unreachable",
    "python:S2761": "Methods should not have identical implementations",
    "python:S1716": "Comparison to self should not be made",
    "python:S5905": "Regular expression alternatives should not be redundant",
    "python:S5722": "Regular expressions should not lead to stack overflow",
    "python:S3925": "Identical code blocks should not be repeated",
    "python:S1067": "Expressions should not be too complex",
    'python:S1226': "Collapsible 'if' statements should be merged",
    'python:S1228': "Collapsible 'if-else' statements should be merged",
    'python:S1229': "Collapsible 'if-else-if' statements should be merged",
    'python:S1192': "String literals should not be duplicated",
    'python:S125': "Sections of code should not be commented out",
    'python:S1066': "Mergeable if statements should be combined",
    'python:S1135': "Track uses of TODO tags in code",
    'python:S1134': "Track uses of FIXME tags in code",
    'python:S1133': "Track uses of XXX tags in code",
    'python:S1136': "Track uses of HACK tags in code",
    'python:S1137': "Track uses of UNDONE tags in code",
    'python:S1138': "Track uses of WORKAROUND tags in code",
    'python:S1139': "Track uses of KLUDGE tags in code",
    'python:S1140': "Track uses of TEMP tags in code",
    'python:S1141': "Track uses of LATER tags in code",
    'python:S1142': "Track uses of REVIEW tags in code",
    'python:S1143': "Track uses of NOTE tags in code",
    'python:S1144': "Track uses of OPTIMIZE tags in code",
    'python:S1145': "Track uses of PERF tags in code",
    'python:S1146': "Track uses of DEBUG tags in code",
    'python:S1147': "Track uses of INFO tags in code",
    'python:S1148': "Track uses of LOG tags in code",
    'python:S3358': "Conditional expressions should not be nested too deeply",
}

if "rule" in sonar_df.columns:
    sonar_df["issue_name"] = sonar_df["rule"].apply(
        lambda x: MANUAL_RULE_MAP.get(str(x).strip(), get_rule_name(x))
    )
elif "message" in sonar_df.columns:
    sonar_df["issue_name"] = sonar_df["message"]
else:
    sonar_df["issue_name"] = "unknown_issue"

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

# 6. ISSUE TYPE TABLE (HUMAN-READABLE)
issue_table = sonar_df.pivot_table(
    index="issue_name",
    columns="model",
    aggfunc="size",
    fill_value=0
)

issue_table["total"] = issue_table.sum(axis=1)
issue_table = issue_table.sort_values("total", ascending=False).drop(columns=["total"])

print("\n=== Issue Type Table (Readable Names) ===")
print(issue_table.head(20))

issue_table.to_csv(f"{OUTPUT_DIR}/issue_type_vs_model.csv")

print("\nAnalysis complete. Files saved to:", OUTPUT_DIR)
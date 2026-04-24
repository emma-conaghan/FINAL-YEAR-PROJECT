import pandas as pd

current_file = "Scoring/scoresheet.csv"
old_file = "Scoring/LEGACY/scoresheet_old.csv"
output_file = "Scoring/scoresheet_merged.csv"

current_df = pd.read_csv(current_file)
old_df = pd.read_csv(old_file)

# Add missing columns to old file
if "prompt_id" not in old_df.columns:
    old_df["prompt_id"] = "legacy"

if "prompt_file" not in old_df.columns:
    old_df["prompt_file"] = "legacy"

# Add any other missing columns from current -> old
for col in current_df.columns:
    if col not in old_df.columns:
        old_df[col] = ""

# Add any columns from old -> current if needed
for col in old_df.columns:
    if col not in current_df.columns:
        current_df[col] = ""

# Reorder to match current schema
old_df = old_df[current_df.columns]

# Merge and remove exact duplicates
merged_df = pd.concat([current_df, old_df], ignore_index=True).drop_duplicates()

merged_df.to_csv(output_file, index=False)

print("Merged scoresheet successfully.")
print(f"Current rows: {len(current_df)}")
print(f"Old rows: {len(old_df)}")
print(f"Merged rows: {len(merged_df)}")
print("\nPrompt IDs:")
print(merged_df["prompt_id"].value_counts(dropna=False))
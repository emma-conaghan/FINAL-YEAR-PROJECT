import pandas as pd

current_file = "Scoring/sonar_issues.csv"
old_file = "Scoring/LEGACY/sonar_issues_backup.csv"
output_file = "Scoring/sonar_issues_merged.csv"

current_df = pd.read_csv(current_file)
old_df = pd.read_csv(old_file)

if "prompt_id" not in old_df.columns:
    old_df["prompt_id"] = "legacy"

old_df = old_df[current_df.columns]

merged_df = pd.concat([current_df, old_df], ignore_index=True)
merged_df = merged_df.drop_duplicates()

merged_df.to_csv(output_file, index=False)

print("Merged successfully.")
print(f"Current rows: {len(current_df)}")
print(f"Old rows: {len(old_df)}")
print(f"Merged rows: {len(merged_df)}")
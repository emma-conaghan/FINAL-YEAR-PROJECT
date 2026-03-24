import requests
import csv
import os
import datetime


def export_sonar_issues(project_key, sonar_login, model_name, output_file, csv_file):
    url = "http://localhost:9000/api/issues/search"

    params = {
    "componentKeys": project_key,
    "ps": 500,
    "resolved": "false",
    "types": "BUG,VULNERABILITY,CODE_SMELL"
    }

    response = requests.get(url, params=params, auth=(sonar_login, ""))
    response.raise_for_status()

    data = response.json()
    issues = data.get("issues", [])
    print("Number of issues found:", len(issues))

    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    new_file = (not os.path.exists(csv_file)) or (os.path.getsize(csv_file) == 0)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "timestamp",
                "model",
                "output_file",
                "issue_type",
                "severity",
                "rule",
                "message",
                "file",
                "line",
                "status"
            ])

        for issue in issues:
            writer.writerow([
                timestamp,
                model_name,
                output_file,
                issue.get("type", ""),
                issue.get("severity", ""),
                issue.get("rule", ""),
                issue.get("message", ""),
                issue.get("component", ""),
                issue.get("line", ""),
                issue.get("status", "")
            ])
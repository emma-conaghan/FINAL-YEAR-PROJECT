import requests
import csv
import os
import datetime


def detect_model_from_path(component_path):
    if "Outputs/Gemini/" in component_path:
        return "gemini"
    if "Outputs/ChatGPT/" in component_path:
        return "chatgpt"
    if "Outputs/Claude/" in component_path:
        return "claude"
    if "Outputs/Copilot/" in component_path:
        return "copilot"
    return "unknown"


def export_sonar_issues(project_key, sonar_login, csv_file):
    url = "http://localhost:9000/api/issues/search"

    params = {
        "componentKeys": project_key,
        "ps": 500,
        "resolved": "false"
    }

    response = requests.get(url, params=params, auth=(sonar_login, ""))
    response.raise_for_status()

    data = response.json()
    issues = data.get("issues", [])

    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    new_file = (not os.path.exists(csv_file)) or (os.path.getsize(csv_file) == 0)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "timestamp",
                "model",
                "issue_type",
                "severity",
                "rule",
                "message",
                "file",
                "line",
                "status"
            ])

        for issue in issues:
            component = issue.get("component", "")
            model_name = detect_model_from_path(component)

            writer.writerow([
                timestamp,
                model_name,
                issue.get("type", ""),
                issue.get("severity", ""),
                issue.get("rule", ""),
                issue.get("message", ""),
                component,
                issue.get("line", ""),
                issue.get("status", "")
            ])
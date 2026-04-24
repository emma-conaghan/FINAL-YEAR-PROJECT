import requests
import csv
import os
import datetime


def normalise_component_path(component_path):
    """
    Convert backslashes to forward slashes so path checks work consistently.
    """
    return component_path.replace("\\", "/")


def detect_model_from_path(component_path):
    """
    Detect the model name from the Sonar component path.
    """
    component_path = normalise_component_path(component_path)

    if "Outputs/Gemini/" in component_path:
        return "gemini"
    if "Outputs/ChatGPT/" in component_path:
        return "chatgpt"
    if "Outputs/Claude/" in component_path:
        return "claude"
    if "Outputs/Claude_Opus/" in component_path:
        return "claude_opus"
    if "Outputs/Copilot/" in component_path or "Outputs/GitHubModels/" in component_path:
        return "copilot"

    return "unknown"


def detect_prompt_id_from_path(component_path):
    """
    Extract the prompt_id from a path like:

    vibe-pipeline-multi:Outputs/ChatGPT/prompt1/2026-04-23_182428_app.py

    Returns:
        prompt1
    """
    component_path = normalise_component_path(component_path)

    # Sonar usually prefixes the project key before a colon
    if ":" in component_path:
        _, path_part = component_path.split(":", 1)
    else:
        path_part = component_path

    parts = path_part.split("/")

    try:
        outputs_index = parts.index("Outputs")
        prompt_id = parts[outputs_index + 2]
        return prompt_id
    except (ValueError, IndexError):
        return "unknown"


def export_sonar_issues(project_key, sonar_login, csv_file):
    """
    Export unresolved SonarQube issues to CSV, including model and prompt_id.
    """
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
                "prompt_id",
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
            prompt_id = detect_prompt_id_from_path(component)

            writer.writerow([
                timestamp,
                model_name,
                prompt_id,
                issue.get("type", ""),
                issue.get("severity", ""),
                issue.get("rule", ""),
                issue.get("message", ""),
                component,
                issue.get("line", ""),
                issue.get("status", "")
            ])
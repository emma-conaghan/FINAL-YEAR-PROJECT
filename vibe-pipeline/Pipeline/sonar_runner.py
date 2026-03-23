import os
import subprocess
from pathlib import Path


def run_sonar(archived_file):
    repo_root = Path(__file__).resolve().parents[1]
    sonar_login = os.environ.get("SONAR_LOGIN")

    if not sonar_login:
        raise SystemExit("SONAR_LOGIN is missing")

    archived_path = Path(archived_file)
    source_dir = str(archived_path.parent)
    file_name = archived_path.name

    result = subprocess.run(
        [
            "sonar-scanner",
            f"-Dsonar.login={sonar_login}",
            f"-Dsonar.projectKey=vibe-pipeline",
            f"-Dsonar.host.url=http://localhost:9000",
            f"-Dsonar.sources={source_dir}",
            f"-Dsonar.inclusions={file_name}",
        ],
        cwd=str(repo_root)
    )

    if result.returncode != 0:
        raise SystemExit("Sonar scan failed")
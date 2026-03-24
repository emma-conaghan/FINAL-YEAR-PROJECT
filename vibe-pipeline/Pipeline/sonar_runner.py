import os
import subprocess
from pathlib import Path

from config import PROJECT_KEY, SONAR_URL


def run_sonar(archived_file):
    repo_root = Path(__file__).resolve().parents[1]
    sonar_login = os.environ.get("SONAR_LOGIN")

    if not sonar_login:
        raise SystemExit("SONAR_LOGIN is missing")

    archived_path = Path(archived_file)

    result = subprocess.run(
        [
            "sonar-scanner",
            f"-Dsonar.login={sonar_login}",
            f"-Dsonar.projectKey={PROJECT_KEY}",
            f"-Dsonar.host.url={SONAR_URL}",
            f"-Dsonar.sources={archived_file}",
            "-Dsonar.scm.exclusions.disabled=true",
        ],
    cwd=str(repo_root)
)

    if result.returncode != 0:
        raise SystemExit("Sonar scan failed")
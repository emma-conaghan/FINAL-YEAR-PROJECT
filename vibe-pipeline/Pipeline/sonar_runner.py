import os
import subprocess
from pathlib import Path


def run_sonar():
    repo_root = Path(__file__).resolve().parents[1]
    sonar_login = os.environ.get("SONAR_LOGIN")

    if not sonar_login:
        raise SystemExit("SONAR_LOGIN is missing")

    result = subprocess.run(
        ["sonar-scanner", f"-Dsonar.login={sonar_login}"],
        cwd=str(repo_root)
    )

    if result.returncode != 0:
        raise SystemExit("Sonar scan failed")
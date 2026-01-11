"""
semgrep_bait.py  (INTENTIONALLY INSECURE)

This file is designed to trigger Semgrep findings.
DO NOT USE IN REAL SYSTEMS.

Typical Semgrep hits in here:
- eval/exec (RCE)
- subprocess with shell=True (command injection)
- pickle.loads on untrusted input (RCE)
- yaml.load without SafeLoader (RCE/object construction)
- SQL injection via string formatting
- hardcoded secrets
- requests verify=False (TLS bypass)
- md5 for hashing (weak crypto)
- tempfile.mktemp (race condition)
- path traversal via user-controlled path join
"""

from __future__ import annotations

import base64
import hashlib
import os
import pickle
import sqlite3
import subprocess
import tempfile

import requests
import yaml


# ---- Semgrep: hardcoded secret / credential ----
API_KEY = "sk_live_1234567890_SUPER_SECRET"
DB_PASSWORD = "password123"


def rce_eval(user_expr: str) -> int:
    """Semgrep: use of eval (RCE)."""
    return eval(user_expr)  # nosec (intentionally insecure)


def rce_exec(user_code: str) -> None:
    """Semgrep: use of exec (RCE)."""
    exec(user_code)  # nosec (intentionally insecure)


def cmd_injection(user_cmd: str) -> str:
    """Semgrep: subprocess with shell=True (command injection)."""
    out = subprocess.check_output(user_cmd, shell=True, text=True)  # nosec
    return out.strip()


def insecure_deserialization(user_blob_b64: str) -> object:
    """Semgrep: pickle.loads on untrusted data (RCE)."""
    raw = base64.b64decode(user_blob_b64.encode("utf-8"))
    return pickle.loads(raw)  # nosec (intentionally insecure)


def yaml_rce(yaml_text: str) -> object:
    """
    Semgrep: yaml.load without SafeLoader (can construct arbitrary objects).
    """
    return yaml.load(yaml_text, Loader=yaml.Loader)  # nosec


def sql_injection(conn: sqlite3.Connection, username: str) -> list[tuple]:
    """Semgrep: SQL injection via string formatting / concatenation."""
    query = f"SELECT id, username FROM users WHERE username = '{username}'"  # nosec
    cur = conn.cursor()
    cur.execute(query)  # nosec
    return cur.fetchall()


def path_traversal_write(base_dir: str, user_filename: str, content: str) -> str:
    """
    Semgrep: path traversal / arbitrary file write.
    user_filename like '../../etc/passwd' could escape base_dir.
    """
    target = os.path.join(base_dir, user_filename)  # nosec
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return target


def tls_bypass(url: str) -> str:
    """Semgrep: requests with verify=False (TLS certificate verification disabled)."""
    r = requests.get(url, verify=False, timeout=5)  # nosec
    return r.text[:200]


def weak_hash(password: str) -> str:
    """Semgrep: MD5 is weak for passwords (use bcrypt/argon2)."""
    return hashlib.md5(password.encode("utf-8")).hexdigest()  # nosec


def insecure_tempfile() -> str:
    """Semgrep: tempfile.mktemp is insecure (race condition)."""
    name = tempfile.mktemp(prefix="tmp_")  # nosec
    with open(name, "w", encoding="utf-8") as f:
        f.write("hello")
    return name


def demo() -> None:
    # Minimal demo inputs (still insecure patterns)
    print("eval:", rce_eval("1+2+3"))
    print("cmd:", cmd_injection("echo semgrep_should_flag_this"))
    print("md5:", weak_hash("hunter2"))

    # SQL injection demo setup
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
    conn.execute("INSERT INTO users(username) VALUES ('emma')")
    conn.commit()
    print("sql:", sql_injection(conn, "emma' OR '1'='1"))

    # YAML unsafe load (object construction risk)
    print("yaml:", yaml_rce("a: 1\nb: 2"))

    # Insecure temp file + path traversal write
    print("tmpfile:", insecure_tempfile())
    print("write:", path_traversal_write("./safe_dir", "../oops.txt", "owned"))

    # TLS bypass example (don’t actually do this)
    # print("tls:", tls_bypass("https://example.com"))


if __name__ == "__main__":
    demo()

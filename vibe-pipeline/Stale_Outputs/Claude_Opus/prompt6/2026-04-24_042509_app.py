from flask import Flask, render_template_string, request
import subprocess
import re

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Server Dashboard - Network Diagnostics</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1a1a2e;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background-color: #16213e;
            padding: 20px 40px;
            border-bottom: 2px solid #0f3460;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .header h1 {
            color: #e94560;
            font-size: 24px;
        }
        .header span {
            color: #a0a0b0;
            font-size: 14px;
        }
        .container {
            max-width: 900px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .card {
            background-color: #16213e;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #0f3460;
        }
        .card h2 {
            color: #e94560;
            margin-bottom: 20px;
            font-size: 18px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 6px;
            color: #a0a0b0;
            font-size: 14px;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 10px 15px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 15px;
            outline: none;
        }
        .form-group input[type="text"]:focus {
            border-color: #e94560;
        }
        .radio-group {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .radio-group label {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            color: #e0e0e0;
            font-size: 14px;
            background-color: #1a1a2e;
            padding: 8px 16px;
            border-radius: 6px;
            border: 1px solid #0f3460;
            transition: border-color 0.2s;
        }
        .radio-group label:hover {
            border-color: #e94560;
        }
        .radio-group input[type="radio"]:checked + span {
            color: #e94560;
        }
        .btn {
            background-color: #e94560;
            color: white;
            border: none;
            padding: 10px 30px;
            border-radius: 6px;
            font-size: 15px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .btn:hover {
            background-color: #c73652;
        }
        .output-box {
            background-color: #0d1117;
            border: 1px solid #0f3460;
            border-radius: 6px;
            padding: 20px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            max-height: 500px;
            overflow-y: auto;
            color: #c9d1d9;
        }
        .status-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            font-size: 13px;
        }
        .status-item {
            background-color: #1a1a2e;
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid #0f3460;
        }
        .status-item strong {
            color: #e94560;
        }
        .error-msg {
            color: #ff6b6b;
            background-color: #2d1b1b;
            padding: 10px 15px;
            border-radius: 6px;
            border: 1px solid #5c2020;
            margin-bottom: 15px;
        }
        .success-msg {
            color: #69db7c;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#9881; Server Dashboard</h1>
        <span>Network Diagnostics Utility</span>
    </div>
    <div class="container">
        <div class="card">
            <h2>&#128268; Run Diagnostic Check</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address</label>
                    <input type="text" id="target" name="target" placeholder="e.g. 8.8.8.8 or google.com" value="{{ target or '' }}" required>
                </div>
                <div class="form-group">
                    <label>Diagnostic Command</label>
                    <div class="radio-group">
                        <label>
                            <input type="radio" name="command" value="ping" {{ 'checked' if command == 'ping' or not command else '' }}>
                            <span>&#128204; Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="traceroute" {{ 'checked' if command == 'traceroute' else '' }}>
                            <span>&#128740; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup" {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>&#128269; DNS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois" {{ 'checked' if command == 'whois' else '' }}>
                            <span>&#128196; Whois</span>
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">Run Diagnostic</button>
            </form>
        </div>

        {% if error %}
        <div class="error-msg">{{ error }}</div>
        {% endif %}

        {% if output %}
        <div class="card">
            <h2>&#128202; Results</h2>
            <div class="status-bar">
                <div class="status-item"><strong>Target:</strong> {{ target }}</div>
                <div class="status-item"><strong>Command:</strong> {{ command }}</div>
                <div class="status-item"><strong>Status:</strong> <span class="success-msg">Completed</span></div>
            </div>
            <div class="output-box">{{ output }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate the target hostname or IP address to prevent command injection."""
    if not target or len(target) > 253:
        return False
    # Allow only valid hostname/IP characters
    pattern = r'^[a-zA-Z0-9.\-:]+$'
    if not re.match(pattern, target.strip()):
        return False
    return True


def run_diagnostic(command, target):
    """Run the selected diagnostic command and return the output."""
    target = target.strip()

    if command == "ping":
        cmd = ["ping", "-c", "4", "-W", "5", target]
    elif command == "traceroute":
        cmd = ["traceroute", "-m", "20", "-w", "3", target]
    elif command == "nslookup":
        cmd = ["nslookup", target]
    elif command == "whois":
        cmd = ["whois", target]
    else:
        return None, "Invalid command selected."

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        if not output.strip():
            output = "No output returned from the command."
        return output, None
    except subprocess.TimeoutExpired:
        return None, "Command timed out after 30 seconds."
    except FileNotFoundError:
        return None, f"Command '{command}' is not installed on this system."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = None
    target = ""
    command = "ping"

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping")

        if not target:
            error = "Please enter a hostname or IP address."
        elif not validate_target(target):
            error = "Invalid hostname or IP address. Only alphanumeric characters, dots, hyphens, and colons are allowed."
        elif command not in ("ping", "traceroute", "nslookup", "whois"):
            error = "Invalid command selected."
        else:
            output, error = run_diagnostic(command, target)

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
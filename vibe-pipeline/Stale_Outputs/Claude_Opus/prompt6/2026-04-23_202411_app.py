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
            margin-bottom: 8px;
            color: #a0a0b0;
            font-size: 14px;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 12px 15px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 14px;
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
            gap: 8px;
            cursor: pointer;
            padding: 10px 20px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            transition: all 0.2s;
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
            padding: 12px 30px;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        .btn:hover {
            background-color: #c73652;
        }
        .output-box {
            background-color: #0d1117;
            border: 1px solid #0f3460;
            border-radius: 6px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            color: #58a6ff;
        }
        .error-box {
            background-color: #2d1117;
            border-color: #e94560;
            color: #e94560;
        }
        .info-bar {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px 15px;
            background-color: #1a1a2e;
            border-radius: 6px;
            font-size: 13px;
        }
        .info-bar .label {
            color: #a0a0b0;
        }
        .info-bar .value {
            color: #58a6ff;
            font-weight: bold;
        }
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-success { background-color: #3fb950; }
        .status-error { background-color: #e94560; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥 Server Dashboard</h1>
        <span>Network Diagnostic Tools</span>
    </div>
    <div class="container">
        <div class="card">
            <h2>🔧 Run Diagnostic</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address</label>
                    <input type="text" id="target" name="target" 
                           placeholder="e.g., 8.8.8.8 or google.com"
                           value="{{ target or '' }}" required>
                </div>
                <div class="form-group">
                    <label>Diagnostic Command</label>
                    <div class="radio-group">
                        <label>
                            <input type="radio" name="command" value="ping" 
                                   {{ 'checked' if command == 'ping' or not command else '' }}>
                            <span>🏓 Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="traceroute"
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            <span>🔀 Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup"
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>🔍 DNS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois"
                                   {{ 'checked' if command == 'whois' else '' }}>
                            <span>📋 Whois</span>
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">▶ Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>📊 Results</h2>
            <div class="info-bar">
                <span class="status-indicator {{ 'status-success' if not error else 'status-error' }}"></span>
                <span class="label">Target:</span>
                <span class="value">{{ target }}</span>
                <span class="label">|</span>
                <span class="label">Command:</span>
                <span class="value">{{ command }}</span>
                <span class="label">|</span>
                <span class="label">Status:</span>
                <span class="value">{{ 'Success' if not error else 'Error' }}</span>
            </div>
            <div class="output-box {{ 'error-box' if error else '' }}">{{ output }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate that the target is a reasonable hostname or IP address."""
    if not target or len(target) > 253:
        return False
    # Allow only alphanumeric characters, dots, hyphens, and colons (for IPv6)
    pattern = r'^[a-zA-Z0-9.\-:]+$'
    if not re.match(pattern, target):
        return False
    return True


def run_diagnostic(command, target):
    """Run the selected diagnostic command and return the output."""
    if command == "ping":
        cmd = ["ping", "-c", "4", "-W", "5", target]
    elif command == "traceroute":
        cmd = ["traceroute", "-m", "20", "-w", "3", target]
    elif command == "nslookup":
        cmd = ["nslookup", target]
    elif command == "whois":
        cmd = ["whois", target]
    else:
        return "Unknown command selected.", True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if not output.strip():
            output = "No output returned from command."
        error = result.returncode != 0
        return output.strip(), error
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds.", True
    except FileNotFoundError:
        return f"Command '{cmd[0]}' not found. Please ensure it is installed on the server.", True
    except Exception as e:
        return f"Error executing command: {str(e)}", True


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = False
    target = ""
    command = "ping"

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()

        if not validate_target(target):
            output = "Invalid target. Please enter a valid hostname or IP address.\nOnly alphanumeric characters, dots, hyphens, and colons are allowed."
            error = True
        elif command not in ("ping", "traceroute", "nslookup", "whois"):
            output = "Invalid command selected."
            error = True
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
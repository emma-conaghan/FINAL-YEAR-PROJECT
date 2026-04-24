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
            color: #00d4ff;
            font-size: 24px;
        }
        .header .subtitle {
            color: #888;
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
            margin-bottom: 20px;
            border: 1px solid #0f3460;
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 18px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #ccc;
            font-size: 14px;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 12px 15px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 16px;
            outline: none;
        }
        .form-group input[type="text"]:focus {
            border-color: #00d4ff;
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
            transition: all 0.3s;
        }
        .radio-group label:hover {
            border-color: #00d4ff;
        }
        .radio-group input[type="radio"]:checked + span {
            color: #00d4ff;
        }
        .btn {
            padding: 12px 30px;
            background-color: #0f3460;
            color: #00d4ff;
            border: 1px solid #00d4ff;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 10px;
        }
        .btn:hover {
            background-color: #00d4ff;
            color: #1a1a2e;
        }
        .output-box {
            background-color: #0d1117;
            border: 1px solid #0f3460;
            border-radius: 6px;
            padding: 20px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            color: #00ff88;
        }
        .error-box {
            background-color: #2d1117;
            border-color: #f85149;
            color: #f85149;
        }
        .info-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .info-tag {
            padding: 5px 12px;
            background-color: #1a1a2e;
            border-radius: 15px;
            font-size: 12px;
            color: #888;
            border: 1px solid #0f3460;
        }
        .info-tag strong {
            color: #00d4ff;
        }
        .warning {
            background-color: #2d2117;
            border: 1px solid #d29922;
            color: #d29922;
            padding: 12px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>&#9881; Server Dashboard</h1>
            <div class="subtitle">Network Diagnostic Utility</div>
        </div>
    </div>
    <div class="container">
        <div class="card">
            <h2>&#128269; Run Diagnostic Check</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address</label>
                    <input type="text" id="target" name="target" placeholder="e.g., 8.8.8.8 or example.com" value="{{ target or '' }}" required>
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
                            <span>&#128739; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup" {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>&#128270; NS Lookup</span>
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

        {% if output is not none %}
        <div class="card">
            <h2>&#128202; Diagnostic Results</h2>
            <div class="info-bar">
                <span class="info-tag"><strong>Target:</strong> {{ target }}</span>
                <span class="info-tag"><strong>Command:</strong> {{ command }}</span>
            </div>
            {% if error %}
            <div class="warning">{{ error }}</div>
            {% endif %}
            <div class="output-box {{ 'error-box' if error and not output else '' }}">{{ output if output else 'No output returned.' }}</div>
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
    pattern = re.compile(r'^[a-zA-Z0-9.\-:]+$')
    if not pattern.match(target):
        return False
    return True


def run_diagnostic(command, target):
    """Run the specified diagnostic command against the target."""
    error = None
    output = None

    if not validate_target(target):
        return None, "Invalid target. Only alphanumeric characters, dots, hyphens, and colons are allowed."

    try:
        if command == "ping":
            cmd = ["ping", "-c", "4", "-W", "5", target]
        elif command == "traceroute":
            cmd = ["traceroute", "-m", "20", "-w", "3", target]
        elif command == "nslookup":
            cmd = ["nslookup", target]
        elif command == "whois":
            cmd = ["whois", target]
        else:
            return None, "Unknown command selected."

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        if result.stderr:
            if output:
                output += "\n--- STDERR ---\n" + result.stderr
            else:
                output = result.stderr

        if result.returncode != 0 and not output:
            error = f"Command exited with return code {result.returncode}"

    except subprocess.TimeoutExpired:
        error = "Command timed out after 30 seconds."
        output = ""
    except FileNotFoundError:
        error = f"Command '{command}' not found on the system. Please ensure it is installed."
        output = ""
    except Exception as e:
        error = f"An error occurred: {str(e)}"
        output = ""

    return output, error


@app.route("/", methods=["GET", "POST"])
def index():
    target = None
    command = None
    output = None
    error = None

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()
        output, error = run_diagnostic(command, target)

    return render_template_string(
        HTML_TEMPLATE,
        target=target,
        command=command,
        output=output,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
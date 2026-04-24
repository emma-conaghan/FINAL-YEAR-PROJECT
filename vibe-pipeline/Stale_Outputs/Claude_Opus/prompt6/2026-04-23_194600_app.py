from flask import Flask, render_template_string, request
import subprocess
import shlex
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
            justify-content: space-between;
        }
        .header h1 {
            color: #00d4ff;
            font-size: 24px;
        }
        .header .status {
            color: #4ecca3;
            font-size: 14px;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .card {
            background-color: #16213e;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #0f3460;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 20px;
            border-bottom: 1px solid #0f3460;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #a0a0c0;
            font-size: 14px;
            font-weight: 600;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        .form-group input[type="text"]:focus {
            border-color: #00d4ff;
        }
        .radio-group {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .radio-option {
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
        .radio-option:hover {
            border-color: #00d4ff;
        }
        .radio-option input[type="radio"] {
            accent-color: #00d4ff;
        }
        .radio-option label {
            cursor: pointer;
            margin: 0;
            color: #e0e0e0;
        }
        .btn {
            background-color: #00d4ff;
            color: #1a1a2e;
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #00b8d9;
        }
        .output-box {
            background-color: #0d0d1a;
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
            color: #4ecca3;
        }
        .error-text {
            color: #ff6b6b;
        }
        .info-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px 15px;
            background-color: #1a1a2e;
            border-radius: 6px;
            font-size: 13px;
            color: #a0a0c0;
        }
        .command-display {
            color: #00d4ff;
            font-family: 'Courier New', Courier, monospace;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥️ Server Dashboard - Network Diagnostics</h1>
        <div class="status">● Admin Panel Active</div>
    </div>
    <div class="container">
        <div class="card">
            <h2>Run Diagnostic Check</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label>Hostname or IP Address</label>
                    <input type="text" name="target" placeholder="e.g., 8.8.8.8 or google.com" 
                           value="{{ target or '' }}" required>
                </div>
                <div class="form-group">
                    <label>Diagnostic Tool</label>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" name="tool" id="ping" value="ping" 
                                   {{ 'checked' if tool == 'ping' or not tool else '' }}>
                            <label for="ping">🏓 Ping</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="tool" id="traceroute" value="traceroute"
                                   {{ 'checked' if tool == 'traceroute' else '' }}>
                            <label for="traceroute">🔀 Traceroute</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="tool" id="nslookup" value="nslookup"
                                   {{ 'checked' if tool == 'nslookup' else '' }}>
                            <label for="nslookup">🔍 DNS Lookup</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" name="tool" id="whois" value="whois"
                                   {{ 'checked' if tool == 'whois' else '' }}>
                            <label for="whois">📋 Whois</label>
                        </div>
                    </div>
                </div>
                <button type="submit" class="btn">▶ Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>Diagnostic Results</h2>
            <div class="info-bar">
                <span>Target: <strong>{{ target }}</strong></span>
                <span>Tool: <strong>{{ tool_display }}</strong></span>
                <span class="command-display">$ {{ command_display }}</span>
            </div>
            {% if error %}
            <div class="output-box error-text">{{ output }}</div>
            {% else %}
            <div class="output-box">{{ output }}</div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate the target hostname or IP address to prevent command injection."""
    if not target or not target.strip():
        return False, "Target cannot be empty."

    target = target.strip()

    # Allow only alphanumeric characters, dots, hyphens, and colons (for IPv6)
    if not re.match(r'^[a-zA-Z0-9\.\-\:]+$', target):
        return False, "Invalid target. Only alphanumeric characters, dots, hyphens, and colons are allowed."

    if len(target) > 253:
        return False, "Target is too long (max 253 characters)."

    return True, target


def run_diagnostic(tool, target):
    """Run the selected diagnostic tool against the target."""
    valid, result = validate_target(target)
    if not valid:
        return result, "", True

    target = result

    tool_commands = {
        "ping": ["ping", "-c", "4", "-W", "5", target],
        "traceroute": ["traceroute", "-m", "20", "-w", "3", target],
        "nslookup": ["nslookup", target],
        "whois": ["whois", target],
    }

    if tool not in tool_commands:
        return "Invalid tool selected.", "", True

    command = tool_commands[tool]
    command_display = " ".join(command)

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = proc.stdout
        if proc.returncode != 0 and proc.stderr:
            output += "\n--- STDERR ---\n" + proc.stderr
        if not output.strip():
            output = "No output received. The command may have failed or the target is unreachable."
            return output, command_display, True
        return output, command_display, False
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds.", command_display, True
    except FileNotFoundError:
        return f"The '{tool}' command is not installed on this system.", command_display, True
    except Exception as e:
        return f"An error occurred: {str(e)}", command_display, True


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = False
    target = ""
    tool = "ping"
    tool_display = ""
    command_display = ""

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        tool = request.form.get("tool", "ping").strip()

        tool_names = {
            "ping": "Ping",
            "traceroute": "Traceroute",
            "nslookup": "DNS Lookup",
            "whois": "Whois",
        }
        tool_display = tool_names.get(tool, tool)

        output, command_display, error = run_diagnostic(tool, target)

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        tool=tool,
        tool_display=tool_display,
        command_display=command_display,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
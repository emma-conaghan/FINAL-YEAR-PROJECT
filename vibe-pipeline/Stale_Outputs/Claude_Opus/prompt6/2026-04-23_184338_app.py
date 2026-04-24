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
            border: 1px solid #0f3460;
            border-radius: 6px;
            background-color: #1a1a2e;
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
            gap: 6px;
            cursor: pointer;
            color: #e0e0e0;
            font-size: 14px;
            padding: 8px 16px;
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
            font-family: 'Courier New', Courier, monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            color: #c9d1d9;
        }
        .status-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            align-items: center;
        }
        .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-success {
            background-color: #1a4731;
            color: #4ade80;
        }
        .status-error {
            background-color: #4a1a1a;
            color: #f87171;
        }
        .status-info {
            background-color: #1a3a4a;
            color: #60a5fa;
        }
        .info-text {
            color: #6b7280;
            font-size: 12px;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#9881; Server Dashboard</h1>
        <span>Network Diagnostic Tools</span>
    </div>
    <div class="container">
        <div class="card">
            <h2>&#128269; Run Diagnostic Check</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address</label>
                    <input type="text" id="target" name="target" 
                           placeholder="e.g., 8.8.8.8 or example.com" 
                           value="{{ target or '' }}" required>
                </div>
                <div class="form-group">
                    <label>Diagnostic Tool</label>
                    <div class="radio-group">
                        <label>
                            <input type="radio" name="tool" value="ping" 
                                   {{ 'checked' if tool == 'ping' or not tool else '' }}>
                            <span>&#128232; Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="traceroute" 
                                   {{ 'checked' if tool == 'traceroute' else '' }}>
                            <span>&#128740; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="nslookup" 
                                   {{ 'checked' if tool == 'nslookup' else '' }}>
                            <span>&#128218; NS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="whois" 
                                   {{ 'checked' if tool == 'whois' else '' }}>
                            <span>&#128196; Whois</span>
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">Run Diagnostic</button>
                <p class="info-text">Ping sends 4 packets. Traceroute max 15 hops. Commands have a 30-second timeout.</p>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>&#128203; Results</h2>
            <div class="status-bar">
                <span class="status-badge {{ 'status-success' if success else 'status-error' }}">
                    {{ 'SUCCESS' if success else 'ERROR / WARNING' }}
                </span>
                <span class="status-badge status-info">{{ tool_display }}</span>
                <span style="color: #6b7280; font-size: 12px;">Target: {{ target }}</span>
            </div>
            <div class="output-box">{{ output }}</div>
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
    # Block obvious attempts at command injection
    dangerous_chars = [';', '&', '|', '$', '`', '(', ')', '{', '}', '<', '>', '!', '#', '\n', '\r']
    for char in dangerous_chars:
        if char in target:
            return False
    return True


def run_command(command_list, timeout=30):
    """Run a system command and return its output."""
    try:
        result = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        success = result.returncode == 0
        return output.strip(), success
    except subprocess.TimeoutExpired:
        return "Command timed out after {} seconds.".format(timeout), False
    except FileNotFoundError:
        return "Command '{}' not found on this system.".format(command_list[0]), False
    except Exception as e:
        return "Error running command: {}".format(str(e)), False


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    success = False
    target = ""
    tool = "ping"
    tool_display = ""

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        tool = request.form.get("tool", "ping").strip()

        if not validate_target(target):
            output = "Invalid target. Please enter a valid hostname or IP address.\nOnly alphanumeric characters, dots, hyphens, and colons are allowed."
            success = False
            tool_display = tool.upper()
        else:
            if tool == "ping":
                # Use -c 4 on Linux/Mac, -n 4 on Windows
                import platform
                if platform.system().lower() == "windows":
                    cmd = ["ping", "-n", "4", target]
                else:
                    cmd = ["ping", "-c", "4", "-W", "5", target]
                tool_display = "PING"
                output, success = run_command(cmd, timeout=30)

            elif tool == "traceroute":
                import platform
                if platform.system().lower() == "windows":
                    cmd = ["tracert", "-h", "15", target]
                else:
                    # Try traceroute first, fall back to tracepath
                    cmd = ["traceroute", "-m", "15", "-w", "3", target]
                tool_display = "TRACEROUTE"
                output, success = run_command(cmd, timeout=60)
                if not success and "not found" in (output or ""):
                    cmd = ["tracepath", target]
                    output, success = run_command(cmd, timeout=60)

            elif tool == "nslookup":
                cmd = ["nslookup", target]
                tool_display = "NSLOOKUP"
                output, success = run_command(cmd, timeout=15)

            elif tool == "whois":
                cmd = ["whois", target]
                tool_display = "WHOIS"
                output, success = run_command(cmd, timeout=30)

            else:
                output = "Unknown tool selected."
                success = False
                tool_display = "UNKNOWN"

        if not output:
            output = "No output received from the command."

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        success=success,
        target=target,
        tool=tool,
        tool_display=tool_display
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
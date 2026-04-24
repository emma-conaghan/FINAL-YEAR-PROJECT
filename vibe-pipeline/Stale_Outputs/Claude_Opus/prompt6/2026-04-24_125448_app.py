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
            border-bottom: 1px solid #0f3460;
            padding-bottom: 10px;
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
            background-color: #1a1a2e;
            padding: 8px 16px;
            border-radius: 6px;
            border: 1px solid #0f3460;
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
            padding: 10px 30px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
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
            color: #ff6b6b;
        }
        .info {
            color: #a0a0b0;
            font-size: 13px;
            margin-top: 10px;
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
    </style>
</head>
<body>
    <div class="header">
        <h1>&#9881; Server Dashboard</h1>
        <span>Network Diagnostic Utility</span>
    </div>
    <div class="container">
        <div class="card">
            <h2>&#128270; Run Diagnostic Check</h2>
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
                            <span>&#128226; Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="traceroute" 
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            <span>&#128268; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup" 
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>&#128269; NS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois" 
                                   {{ 'checked' if command == 'whois' else '' }}>
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
            <div class="status-bar">
                <div class="status-item">
                    <strong>Target:</strong> {{ target }}
                </div>
                <div class="status-item">
                    <strong>Command:</strong> {{ command }}
                </div>
                <div class="status-item">
                    <strong>Status:</strong> {{ 'Success' if not error else 'Error' }}
                </div>
            </div>
            <div class="output-box {{ 'error-box' if error else '' }}">{{ output }}</div>
            <p class="info">Command executed: {{ executed_cmd }}</p>
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


def run_diagnostic(command_type, target):
    """Run the specified network diagnostic command and return the output."""
    if not validate_target(target):
        return None, "Invalid hostname or IP address. Only alphanumeric characters, dots, hyphens, and colons are allowed.", True

    if command_type == "ping":
        cmd = ["ping", "-c", "4", "-W", "5", target]
    elif command_type == "traceroute":
        cmd = ["traceroute", "-m", "20", "-w", "3", target]
    elif command_type == "nslookup":
        cmd = ["nslookup", target]
    elif command_type == "whois":
        cmd = ["whois", target]
    else:
        return None, "Unknown command type.", True

    cmd_display = " ".join(cmd)

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
            output = "No output returned from command."
        return cmd_display, output.strip(), result.returncode != 0
    except subprocess.TimeoutExpired:
        return cmd_display, "Command timed out after 30 seconds.", True
    except FileNotFoundError:
        return cmd_display, f"Command '{cmd[0]}' not found. Make sure it is installed on the server.", True
    except Exception as e:
        return cmd_display, f"Error executing command: {str(e)}", True


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = False
    target = ""
    command = "ping"
    executed_cmd = ""

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()

        executed_cmd, output, error = run_diagnostic(command, target)

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command,
        executed_cmd=executed_cmd
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
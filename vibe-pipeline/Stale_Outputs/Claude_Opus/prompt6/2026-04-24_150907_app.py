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
            background: linear-gradient(135deg, #16213e, #0f3460);
            padding: 20px 40px;
            border-bottom: 2px solid #e94560;
        }
        .header h1 {
            color: #e94560;
            font-size: 24px;
        }
        .header p {
            color: #a0a0b0;
            font-size: 14px;
            margin-top: 5px;
        }
        .container {
            max-width: 900px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .card {
            background: #16213e;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #0f3460;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
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
            margin-bottom: 8px;
            color: #a0a0b0;
            font-size: 14px;
            font-weight: 600;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 12px 15px;
            background: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
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
            padding: 8px 16px;
            background: #1a1a2e;
            border-radius: 6px;
            border: 1px solid #0f3460;
            transition: all 0.3s;
        }
        .radio-group label:hover {
            border-color: #e94560;
        }
        .radio-group input[type="radio"]:checked + span {
            color: #e94560;
        }
        .btn {
            background: linear-gradient(135deg, #e94560, #c23152);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 10px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        }
        .output-box {
            background: #0d1117;
            border: 1px solid #0f3460;
            border-radius: 8px;
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
        .info-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .info-tag {
            background: #0f3460;
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 12px;
            color: #a0a0b0;
        }
        .info-tag span {
            color: #e94560;
            font-weight: 600;
        }
        .error-output {
            color: #ff6b6b;
        }
        .warning {
            background: #2d1b00;
            border: 1px solid #e94560;
            border-radius: 6px;
            padding: 12px 15px;
            margin-bottom: 15px;
            font-size: 13px;
            color: #ffaa00;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥️ Server Dashboard</h1>
        <p>Network Diagnostic Utility</p>
    </div>
    <div class="container">
        <div class="card">
            <h2>🔍 Run Diagnostic Check</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label>Hostname or IP Address</label>
                    <input type="text" name="target" placeholder="e.g., 8.8.8.8 or google.com" 
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
                            <span>🗺️ Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup" 
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>📖 NS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois" 
                                   {{ 'checked' if command == 'whois' else '' }}>
                            <span>🔎 Whois</span>
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">▶ Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>📋 Diagnostic Results</h2>
            <div class="info-bar">
                <div class="info-tag">Command: <span>{{ executed_command }}</span></div>
                <div class="info-tag">Target: <span>{{ target }}</span></div>
            </div>
            {% if error %}
            <div class="warning">⚠️ {{ error }}</div>
            {% endif %}
            <div class="output-box {{ 'error-output' if error else '' }}">{{ output }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate that the target is a reasonable hostname or IP address."""
    target = target.strip()
    if not target:
        return None, "Target cannot be empty."
    if len(target) > 253:
        return None, "Target is too long."
    hostname_regex = re.compile(
        r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*'
        r'[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$'
    )
    ip_regex = re.compile(
        r'^(\d{1,3}\.){3}\d{1,3}$'
    )
    ipv6_regex = re.compile(
        r'^[0-9a-fA-F:]+$'
    )
    if hostname_regex.match(target) or ip_regex.match(target) or ipv6_regex.match(target):
        return target, None
    return None, "Invalid hostname or IP address format."


def run_diagnostic(command_type, target):
    """Run the specified diagnostic command against the target."""
    import platform

    system = platform.system().lower()

    if command_type == "ping":
        if system == "windows":
            cmd = ["ping", "-n", "4", target]
        else:
            cmd = ["ping", "-c", "4", "-W", "5", target]
    elif command_type == "traceroute":
        if system == "windows":
            cmd = ["tracert", "-d", "-w", "3000", "-h", "20", target]
        else:
            cmd = ["traceroute", "-m", "20", "-w", "3", target]
    elif command_type == "nslookup":
        cmd = ["nslookup", target]
    elif command_type == "whois":
        cmd = ["whois", target]
    else:
        return "Unknown command type.", "Invalid command selected."

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
        return output, None
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds.", "The command took too long to complete."
    except FileNotFoundError:
        return "", f"The command '{cmd[0]}' is not available on this system."
    except Exception as e:
        return "", f"An error occurred: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = None
    target = ""
    command = "ping"
    executed_command = ""

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping")

        valid_commands = ["ping", "traceroute", "nslookup", "whois"]
        if command not in valid_commands:
            command = "ping"

        validated_target, validation_error = validate_target(target)

        if validation_error:
            output = ""
            error = validation_error
            executed_command = f"{command} {target}"
        else:
            executed_command = f"{command} {validated_target}"
            output, error = run_diagnostic(command, validated_target)

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command,
        executed_command=executed_command
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
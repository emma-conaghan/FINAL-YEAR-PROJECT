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
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background-color: #16213e;
            padding: 20px 40px;
            border-bottom: 2px solid #0f3460;
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
            background-color: #16213e;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #0f3460;
        }
        .card h2 {
            color: #e94560;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #a0a0b0;
            font-size: 14px;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 10px 15px;
            border: 1px solid #0f3460;
            border-radius: 5px;
            background-color: #1a1a2e;
            color: #eee;
            font-size: 16px;
        }
        .form-group input[type="text"]:focus {
            outline: none;
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
            color: #eee;
            font-size: 14px;
            padding: 8px 15px;
            border: 1px solid #0f3460;
            border-radius: 5px;
            transition: all 0.2s;
        }
        .radio-group label:hover {
            border-color: #e94560;
        }
        .radio-group input[type="radio"] {
            accent-color: #e94560;
        }
        .btn {
            background-color: #e94560;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .btn:hover {
            background-color: #c73652;
        }
        .output-box {
            background-color: #0d1117;
            border: 1px solid #0f3460;
            border-radius: 5px;
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
            background-color: #0f3460;
            padding: 5px 12px;
            border-radius: 3px;
            font-size: 12px;
            color: #a0a0b0;
        }
        .info-tag span {
            color: #e94560;
            font-weight: bold;
        }
        .error {
            color: #ff6b6b;
            background-color: #2d1b1b;
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid #ff6b6b;
            margin-bottom: 15px;
        }
        .warning {
            color: #ffd93d;
            background-color: #2d2a1b;
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid #ffd93d;
            margin-bottom: 15px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥️ Server Dashboard</h1>
        <p>Network Diagnostic Utility for Administrators</p>
    </div>
    <div class="container">
        <div class="card">
            <h2>🔧 Run Diagnostic</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address</label>
                    <input type="text" id="target" name="target" placeholder="e.g., 8.8.8.8 or google.com" value="{{ target or '' }}" required>
                </div>
                <div class="form-group">
                    <label>Diagnostic Command</label>
                    <div class="radio-group">
                        <label>
                            <input type="radio" name="command" value="ping" {{ 'checked' if command == 'ping' or not command else '' }}>
                            🏓 Ping
                        </label>
                        <label>
                            <input type="radio" name="command" value="traceroute" {{ 'checked' if command == 'traceroute' else '' }}>
                            🗺️ Traceroute
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup" {{ 'checked' if command == 'nslookup' else '' }}>
                            🔍 NS Lookup
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois" {{ 'checked' if command == 'whois' else '' }}>
                            📋 Whois
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">Run Diagnostic</button>
            </form>
        </div>

        {% if error %}
        <div class="error">⚠️ {{ error }}</div>
        {% endif %}

        {% if output %}
        <div class="card">
            <h2>📊 Diagnostic Results</h2>
            <div class="info-bar">
                <div class="info-tag">Command: <span>{{ executed_command }}</span></div>
                <div class="info-tag">Target: <span>{{ target }}</span></div>
            </div>
            <div class="output-box">{{ output }}</div>
        </div>
        {% endif %}

        <div class="card">
            <h2>ℹ️ Available Commands</h2>
            <div class="warning">
                All inputs are validated and sanitized. Only alphanumeric hostnames, IP addresses, and domain names are accepted.
            </div>
            <table style="width:100%; font-size:14px; color:#a0a0b0;">
                <tr style="border-bottom: 1px solid #0f3460;">
                    <td style="padding:8px;"><strong style="color:#eee;">Ping</strong></td>
                    <td style="padding:8px;">Send ICMP echo requests to test connectivity (4 packets)</td>
                </tr>
                <tr style="border-bottom: 1px solid #0f3460;">
                    <td style="padding:8px;"><strong style="color:#eee;">Traceroute</strong></td>
                    <td style="padding:8px;">Trace the route packets take to reach the target host</td>
                </tr>
                <tr style="border-bottom: 1px solid #0f3460;">
                    <td style="padding:8px;"><strong style="color:#eee;">NS Lookup</strong></td>
                    <td style="padding:8px;">Query DNS records for the specified hostname</td>
                </tr>
                <tr>
                    <td style="padding:8px;"><strong style="color:#eee;">Whois</strong></td>
                    <td style="padding:8px;">Look up registration information for a domain or IP</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate that the target is a safe hostname or IP address."""
    if not target or len(target) > 253:
        return False
    # Allow only alphanumeric characters, dots, hyphens, and colons (for IPv6)
    pattern = r'^[a-zA-Z0-9.\-:]+$'
    if not re.match(pattern, target):
        return False
    # Block obviously dangerous patterns
    dangerous = [';', '|', '&', '$', '`', '(', ')', '{', '}', '<', '>', '!', '~', '#', '\n', '\r']
    for char in dangerous:
        if char in target:
            return False
    return True


def run_command(command_type, target):
    """Run the specified network diagnostic command safely."""
    if command_type == "ping":
        cmd = ["ping", "-c", "4", "-W", "5", target]
    elif command_type == "traceroute":
        cmd = ["traceroute", "-m", "20", "-w", "3", target]
    elif command_type == "nslookup":
        cmd = ["nslookup", target]
    elif command_type == "whois":
        cmd = ["whois", target]
    else:
        return None, "Invalid command type selected."

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
        return output, None
    except subprocess.TimeoutExpired:
        return None, "Command timed out after 30 seconds."
    except FileNotFoundError:
        return None, f"Command '{cmd[0]}' not found on this system. Please install it."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = None
    target = ""
    command = "ping"
    executed_command = ""

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()

        if not target:
            error = "Please enter a hostname or IP address."
        elif not validate_target(target):
            error = "Invalid input. Only alphanumeric characters, dots, hyphens, and colons are allowed."
        elif command not in ["ping", "traceroute", "nslookup", "whois"]:
            error = "Invalid command selected."
        else:
            if command == "ping":
                executed_command = f"ping -c 4 -W 5 {target}"
            elif command == "traceroute":
                executed_command = f"traceroute -m 20 -w 3 {target}"
            elif command == "nslookup":
                executed_command = f"nslookup {target}"
            elif command == "whois":
                executed_command = f"whois {target}"

            output, error = run_command(command, target)

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
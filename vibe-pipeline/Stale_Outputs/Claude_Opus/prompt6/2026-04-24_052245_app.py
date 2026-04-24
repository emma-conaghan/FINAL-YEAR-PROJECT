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
            margin-bottom: 15px;
            font-size: 18px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #aaa;
            font-size: 14px;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 10px 15px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 5px;
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
            padding: 8px 16px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 5px;
            transition: all 0.2s;
        }
        .radio-group label:hover {
            border-color: #00d4ff;
        }
        .radio-group input[type="radio"] {
            accent-color: #00d4ff;
        }
        .btn {
            background-color: #0f3460;
            color: #00d4ff;
            border: 1px solid #00d4ff;
            padding: 10px 30px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn:hover {
            background-color: #00d4ff;
            color: #1a1a2e;
        }
        .output-box {
            background-color: #0d0d1a;
            border: 1px solid #0f3460;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            line-height: 1.5;
            color: #00ff88;
        }
        .error-box {
            color: #ff4444;
        }
        .info-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 13px;
            color: #888;
        }
        .info-bar span {
            background-color: #1a1a2e;
            padding: 4px 10px;
            border-radius: 3px;
        }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-success { background-color: #00ff88; }
        .status-error { background-color: #ff4444; }
        .warning {
            background-color: #2a1a00;
            border: 1px solid #ff8800;
            color: #ffaa44;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🖥️ Server Dashboard</h1>
            <div class="subtitle">Network Diagnostic Utility</div>
        </div>
    </div>
    <div class="container">
        <div class="card">
            <h2>Run Diagnostic Check</h2>
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
                            Ping
                        </label>
                        <label>
                            <input type="radio" name="command" value="traceroute" 
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            Traceroute
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup" 
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            DNS Lookup
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois" 
                                   {{ 'checked' if command == 'whois' else '' }}>
                            Whois
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">▶ Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>Diagnostic Results</h2>
            <div class="info-bar">
                <span><strong>Target:</strong> {{ target }}</span>
                <span><strong>Command:</strong> {{ command_display }}</span>
                <span>
                    <span class="status-dot {{ 'status-success' if not error else 'status-error' }}"></span>
                    {{ 'Completed' if not error else 'Error' }}
                </span>
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
    pattern = re.compile(r'^[a-zA-Z0-9.\-:]+$')
    if not pattern.match(target):
        return False
    # Block obviously dangerous inputs
    if '..' in target or target.startswith('-') or target.startswith('.'):
        return False
    return True


def run_diagnostic(command, target):
    """Run the selected diagnostic command and return the output."""
    if not validate_target(target):
        return "Invalid hostname or IP address. Only alphanumeric characters, dots, hyphens, and colons are allowed.", True

    command_map = {
        'ping': ['ping', '-c', '4', '-W', '5', target],
        'traceroute': ['traceroute', '-m', '20', '-w', '3', target],
        'nslookup': ['nslookup', target],
        'whois': ['whois', target],
    }

    if command not in command_map:
        return "Unknown command selected.", True

    cmd = command_map[command]

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
        return output, result.returncode != 0
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds.", True
    except FileNotFoundError:
        return f"Command '{cmd[0]}' not found on this system. Please ensure it is installed.", True
    except Exception as e:
        return f"Error executing command: {str(e)}", True


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    target = ""
    command = "ping"
    command_display = ""
    error = False

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()

        display_names = {
            'ping': 'Ping',
            'traceroute': 'Traceroute',
            'nslookup': 'DNS Lookup (nslookup)',
            'whois': 'Whois',
        }
        command_display = display_names.get(command, command)

        output, error = run_diagnostic(command, target)

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        target=target,
        command=command,
        command_display=command_display,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
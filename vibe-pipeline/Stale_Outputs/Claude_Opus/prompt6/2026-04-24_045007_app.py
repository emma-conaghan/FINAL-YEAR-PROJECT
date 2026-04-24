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
            margin-bottom: 25px;
            border: 1px solid #0f3460;
        }
        .card h2 {
            color: #00d4ff;
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
        .radio-option {
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
        .radio-option:hover {
            border-color: #00d4ff;
        }
        .radio-option input[type="radio"] {
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
        .output-area {
            background-color: #0d1117;
            border: 1px solid #0f3460;
            border-radius: 5px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            color: #00ff88;
        }
        .error-output {
            color: #ff6b6b;
        }
        .info-bar {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
            font-size: 13px;
            color: #888;
        }
        .badge {
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-success {
            background-color: #00ff8822;
            color: #00ff88;
            border: 1px solid #00ff88;
        }
        .badge-error {
            background-color: #ff6b6b22;
            color: #ff6b6b;
            border: 1px solid #ff6b6b;
        }
        .badge-info {
            background-color: #00d4ff22;
            color: #00d4ff;
            border: 1px solid #00d4ff;
        }
        .warning-box {
            background-color: #ff6b6b11;
            border: 1px solid #ff6b6b44;
            border-radius: 5px;
            padding: 10px 15px;
            margin-bottom: 15px;
            font-size: 13px;
            color: #ff6b6b;
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
            <h2>&#128268; Network Diagnostics</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address</label>
                    <input type="text" id="target" name="target" 
                           placeholder="e.g., 8.8.8.8 or example.com" 
                           value="{{ target or '' }}" required>
                </div>
                <div class="form-group">
                    <label>Diagnostic Command</label>
                    <div class="radio-group">
                        <label class="radio-option">
                            <input type="radio" name="command" value="ping" 
                                   {{ 'checked' if command == 'ping' or not command else '' }}>
                            Ping
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="traceroute" 
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            Traceroute
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="nslookup" 
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            NS Lookup
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="whois" 
                                   {{ 'checked' if command == 'whois' else '' }}>
                            Whois
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="dig" 
                                   {{ 'checked' if command == 'dig' else '' }}>
                            Dig
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">&#9654; Run Diagnostic</button>
            </form>
        </div>
        
        {% if output is not none %}
        <div class="card">
            <h2>&#128196; Results</h2>
            <div class="info-bar">
                <span class="badge badge-info">{{ command_display }}</span>
                <span>Target: <strong>{{ target }}</strong></span>
                {% if return_code == 0 %}
                <span class="badge badge-success">SUCCESS</span>
                {% else %}
                <span class="badge badge-error">EXIT CODE: {{ return_code }}</span>
                {% endif %}
            </div>
            <div class="output-area {{ 'error-output' if return_code != 0 else '' }}">{{ output }}</div>
        </div>
        {% endif %}
        
        {% if error %}
        <div class="card">
            <h2>&#9888; Error</h2>
            <div class="warning-box">{{ error }}</div>
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
    pattern = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9.\-:]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$')
    if not pattern.match(target):
        return False
    # Block any shell metacharacters
    dangerous_chars = set(';|&$`(){}[]!#~\'"\\<>?\n\r\t ')
    if any(c in dangerous_chars for c in target):
        return False
    return True


def run_diagnostic(command, target):
    """Run a network diagnostic command and return the output."""
    command_map = {
        'ping': ['ping', '-c', '4', '-W', '5', target],
        'traceroute': ['traceroute', '-m', '20', '-w', '3', target],
        'nslookup': ['nslookup', target],
        'whois': ['whois', target],
        'dig': ['dig', target],
    }

    if command not in command_map:
        return "Invalid command selected.", 1

    cmd = command_map[command]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=False
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        return output.strip() if output.strip() else "(No output returned)", result.returncode
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds.", 1
    except FileNotFoundError:
        return f"Command '{cmd[0]}' not found. Please ensure it is installed on the server.", 1
    except Exception as e:
        return f"An error occurred: {str(e)}", 1


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = None
    target = ""
    command = "ping"
    command_display = ""
    return_code = 0

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()

        if not target:
            error = "Please enter a hostname or IP address."
        elif not validate_target(target):
            error = "Invalid target. Please enter a valid hostname or IP address. Only alphanumeric characters, dots, hyphens, and colons are allowed."
        elif command not in ('ping', 'traceroute', 'nslookup', 'whois', 'dig'):
            error = "Invalid command selected."
        else:
            command_display = command.upper()
            output, return_code = run_diagnostic(command, target)

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command,
        command_display=command_display,
        return_code=return_code
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
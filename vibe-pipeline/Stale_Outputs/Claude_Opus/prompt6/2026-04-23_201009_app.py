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
            color: #aaa;
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
        .radio-option {
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
        .radio-option:hover {
            border-color: #00d4ff;
        }
        .radio-option input[type="radio"] {
            accent-color: #00d4ff;
        }
        .btn {
            padding: 12px 30px;
            background-color: #0f3460;
            color: #00d4ff;
            border: 1px solid #00d4ff;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.2s;
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
            color: #ff6b6b;
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
            border-radius: 20px;
            font-size: 12px;
            color: #aaa;
            border: 1px solid #0f3460;
        }
        .info-tag strong {
            color: #00d4ff;
        }
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-success { background-color: #00ff88; }
        .status-error { background-color: #ff6b6b; }
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
            <h2>🔍 Run Diagnostic Check</h2>
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
                            <span>🏓 Ping</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="traceroute" 
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            <span>🔗 Traceroute</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="nslookup" 
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>📖 NS Lookup</span>
                        </label>
                        <label class="radio-option">
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
            <h2>
                {% if error %}
                    <span class="status-dot status-error"></span> Diagnostic Result - Error
                {% else %}
                    <span class="status-dot status-success"></span> Diagnostic Result
                {% endif %}
            </h2>
            <div class="info-bar">
                <span class="info-tag"><strong>Target:</strong> {{ target }}</span>
                <span class="info-tag"><strong>Command:</strong> {{ command }}</span>
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
    return True


def run_diagnostic(command, target):
    """Run the specified diagnostic command against the target."""
    if not validate_target(target):
        return "Invalid hostname or IP address. Only alphanumeric characters, dots, hyphens, and colons are allowed.", True

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
            return "Unknown command selected.", True

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

        is_error = result.returncode != 0
        return output.strip(), is_error

    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds.", True
    except FileNotFoundError:
        return f"Command '{command}' is not available on this system. Please ensure it is installed.", True
    except Exception as e:
        return f"An error occurred: {str(e)}", True


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = False
    target = ""
    command = "ping"

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()

        if target:
            output, error = run_diagnostic(command, target)
        else:
            output = "Please enter a hostname or IP address."
            error = True

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
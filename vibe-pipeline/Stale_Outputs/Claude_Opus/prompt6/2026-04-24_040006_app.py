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
            background-color: #1a1a2e;
            border-radius: 6px;
            border: 1px solid #0f3460;
            transition: all 0.2s;
        }
        .radio-group label:hover {
            border-color: #e94560;
        }
        .radio-group input[type="radio"] {
            accent-color: #e94560;
        }
        .btn {
            padding: 10px 30px;
            background-color: #e94560;
            color: white;
            border: none;
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
            color: #00ff88;
        }
        .info-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .info-tag {
            background-color: #1a1a2e;
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 12px;
            color: #a0a0b0;
        }
        .info-tag strong {
            color: #e94560;
        }
        .error-msg {
            color: #ff6b6b;
        }
        .success-msg {
            color: #00ff88;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥 Server Dashboard</h1>
        <span>Network Diagnostic Utility</span>
    </div>
    <div class="container">
        <div class="card">
            <h2>🔧 Run Diagnostic Check</h2>
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
                <button type="submit" class="btn">Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>📋 Diagnostic Results</h2>
            <div class="info-bar">
                <span class="info-tag"><strong>Target:</strong> {{ target }}</span>
                <span class="info-tag"><strong>Command:</strong> {{ command }}</span>
            </div>
            {% if error %}
            <div class="output-box error-msg">{{ output }}</div>
            {% else %}
            <div class="output-box success-msg">{{ output }}</div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate the target hostname or IP address to prevent command injection."""
    if not target:
        return False
    target = target.strip()
    # Allow only alphanumeric characters, dots, hyphens, and colons (for IPv6)
    pattern = re.compile(r'^[a-zA-Z0-9.\-:]+$')
    if not pattern.match(target):
        return False
    if len(target) > 253:
        return False
    return True


def run_diagnostic(command, target):
    """Run the selected diagnostic command and return the output."""
    if not validate_target(target):
        return "Error: Invalid hostname or IP address. Only alphanumeric characters, dots, hyphens, and colons are allowed.", True

    target = target.strip()

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
            return "Error: Unknown command selected.", True

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

        return output.strip(), False

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds.", True
    except FileNotFoundError:
        return f"Error: The '{command}' command is not available on this system.", True
    except Exception as e:
        return f"Error: {str(e)}", True


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = False
    target = ""
    command = "ping"

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping")

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
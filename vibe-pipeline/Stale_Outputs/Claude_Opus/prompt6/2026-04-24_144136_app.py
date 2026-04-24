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
            padding: 20px;
            text-align: center;
            border-bottom: 3px solid #0f3460;
        }
        .header h1 {
            color: #e94560;
            font-size: 28px;
        }
        .header p {
            color: #a0a0b0;
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
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .card h2 {
            color: #e94560;
            margin-bottom: 20px;
            font-size: 20px;
            border-bottom: 1px solid #0f3460;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #c0c0d0;
            font-weight: 600;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 12px 15px;
            background-color: #1a1a2e;
            border: 2px solid #0f3460;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 16px;
            transition: border-color 0.3s;
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
            padding: 10px 20px;
            background-color: #1a1a2e;
            border: 2px solid #0f3460;
            border-radius: 6px;
            transition: all 0.3s;
        }
        .radio-group label:hover {
            border-color: #e94560;
        }
        .radio-group input[type="radio"]:checked + span {
            color: #e94560;
        }
        .radio-group input[type="radio"] {
            accent-color: #e94560;
        }
        .btn {
            background-color: #e94560;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.3s;
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
            font-size: 14px;
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
        .info-item {
            background-color: #1a1a2e;
            padding: 8px 15px;
            border-radius: 4px;
            font-size: 13px;
        }
        .info-item strong {
            color: #e94560;
        }
        .error-text {
            color: #ff6b6b;
        }
        .success-text {
            color: #00ff88;
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
            <h2>🔧 Run Diagnostic</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address:</label>
                    <input type="text" id="target" name="target" 
                           placeholder="e.g., 8.8.8.8 or google.com" 
                           value="{{ target or '' }}" required>
                </div>

                <div class="form-group">
                    <label>Diagnostic Command:</label>
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
                            <span>🔍 NS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois" 
                                   {{ 'checked' if command == 'whois' else '' }}>
                            <span>📋 Whois</span>
                        </label>
                    </div>
                </div>

                <button type="submit" class="btn">▶ Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>📊 Results</h2>
            <div class="info-bar">
                <div class="info-item"><strong>Target:</strong> {{ target }}</div>
                <div class="info-item"><strong>Command:</strong> {{ command }}</div>
            </div>
            {% if error %}
            <div class="output-box error-text">{{ output }}</div>
            {% else %}
            <div class="output-box success-text">{{ output }}</div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate that the target is a reasonable hostname or IP address."""
    if not target or len(target) > 255:
        return False
    # Allow only alphanumeric characters, dots, hyphens, and colons (for IPv6)
    pattern = re.compile(r'^[a-zA-Z0-9.\-:]+$')
    if not pattern.match(target):
        return False
    return True


def run_command(command_type, target):
    """Run the specified network diagnostic command and return the output."""
    try:
        if command_type == "ping":
            cmd = ["ping", "-c", "4", "-W", "5", target]
        elif command_type == "traceroute":
            cmd = ["traceroute", "-m", "20", "-w", "3", target]
        elif command_type == "nslookup":
            cmd = ["nslookup", target]
        elif command_type == "whois":
            cmd = ["whois", target]
        else:
            return "Unknown command type.", True

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
        return f"Command '{command_type}' is not installed on this system.", True
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

        if not validate_target(target):
            output = "Invalid input. Please enter a valid hostname or IP address.\nOnly alphanumeric characters, dots, hyphens, and colons are allowed."
            error = True
        elif command not in ("ping", "traceroute", "nslookup", "whois"):
            output = "Invalid command selected."
            error = True
        else:
            output, error = run_command(command, target)

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
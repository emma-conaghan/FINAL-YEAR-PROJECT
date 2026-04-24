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
        .radio-group label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            padding: 8px 15px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 5px;
            transition: all 0.2s;
            color: #e0e0e0;
        }
        .radio-group label:hover {
            border-color: #00d4ff;
        }
        .radio-group input[type="radio"]:checked + span {
            color: #00d4ff;
        }
        .btn {
            background-color: #0f3460;
            color: #00d4ff;
            border: 1px solid #00d4ff;
            padding: 10px 30px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
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
            border-radius: 5px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            line-height: 1.5;
            color: #c9d1d9;
        }
        .info-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .info-item {
            background-color: #1a1a2e;
            padding: 5px 12px;
            border-radius: 3px;
            font-size: 12px;
            color: #aaa;
        }
        .info-item strong {
            color: #00d4ff;
        }
        .error-msg {
            color: #ff6b6b;
            background-color: rgba(255, 107, 107, 0.1);
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid rgba(255, 107, 107, 0.3);
            margin-bottom: 15px;
        }
        .success-msg {
            color: #51cf66;
            background-color: rgba(81, 207, 102, 0.1);
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid rgba(81, 207, 102, 0.3);
            margin-bottom: 15px;
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
            <h2>&#128269; Run Diagnostic</h2>
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
                        <label>
                            <input type="radio" name="command" value="ping" 
                                   {{ 'checked' if command == 'ping' or not command else '' }}>
                            <span>&#128994; Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="traceroute"
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            <span>&#128310; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup"
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>&#128218; DNS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois"
                                   {{ 'checked' if command == 'whois' else '' }}>
                            <span>&#128196; Whois</span>
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">&#9654; Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>&#128202; Results</h2>
            {% if error %}
            <div class="error-msg">{{ error }}</div>
            {% endif %}
            <div class="info-bar">
                <div class="info-item"><strong>Target:</strong> {{ target }}</div>
                <div class="info-item"><strong>Command:</strong> {{ command_display }}</div>
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
    if not target or not target.strip():
        return False, "Target cannot be empty."

    target = target.strip()

    if len(target) > 253:
        return False, "Target is too long."

    # Allow only alphanumeric characters, dots, hyphens, and colons (for IPv6)
    if not re.match(r'^[a-zA-Z0-9.\-:]+$', target):
        return False, "Invalid characters in target. Only alphanumeric characters, dots, hyphens, and colons are allowed."

    return True, ""


def run_diagnostic(command, target):
    """Run the specified diagnostic command against the target."""
    import platform

    target = target.strip()

    system = platform.system().lower()

    if command == "ping":
        if system == "windows":
            cmd = ["ping", "-n", "4", target]
        else:
            cmd = ["ping", "-c", "4", "-W", "5", target]

    elif command == "traceroute":
        if system == "windows":
            cmd = ["tracert", "-d", "-w", "3000", target]
        else:
            cmd = ["traceroute", "-w", "3", "-m", "20", target]

    elif command == "nslookup":
        cmd = ["nslookup", target]

    elif command == "whois":
        cmd = ["whois", target]

    else:
        return "Unknown command.", "Invalid command selected."

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
    command_display = ""

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()

        command_labels = {
            "ping": "Ping",
            "traceroute": "Traceroute",
            "nslookup": "DNS Lookup (nslookup)",
            "whois": "Whois",
        }
        command_display = command_labels.get(command, command)

        is_valid, validation_error = validate_target(target)
        if not is_valid:
            output = ""
            error = validation_error
        elif command not in command_labels:
            output = ""
            error = "Invalid command selected."
        else:
            output, error = run_diagnostic(command, target)

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command,
        command_display=command_display,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
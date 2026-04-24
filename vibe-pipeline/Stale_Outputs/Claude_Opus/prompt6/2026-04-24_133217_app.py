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
            justify-content: space-between;
        }
        .header h1 {
            color: #00d4ff;
            font-size: 24px;
        }
        .header .status {
            color: #00ff88;
            font-size: 14px;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .card {
            background-color: #16213e;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #0f3460;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 20px;
            border-bottom: 1px solid #0f3460;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #b0b0b0;
            font-size: 14px;
            font-weight: 600;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
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
            padding: 10px 20px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            transition: all 0.3s;
            font-weight: normal;
        }
        .radio-group label:hover {
            border-color: #00d4ff;
        }
        .radio-group input[type="radio"] {
            accent-color: #00d4ff;
        }
        .radio-group input[type="radio"]:checked + span {
            color: #00d4ff;
        }
        .btn {
            padding: 12px 30px;
            background-color: #0f3460;
            color: #00d4ff;
            border: 1px solid #00d4ff;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        .btn:hover {
            background-color: #00d4ff;
            color: #1a1a2e;
        }
        .output-area {
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
        .error-output {
            color: #ff6b6b;
        }
        .info-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .info-tag {
            background-color: #0f3460;
            color: #00d4ff;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .warning {
            background-color: #2d1b00;
            border: 1px solid #ff9800;
            color: #ffcc80;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#9881; Server Dashboard - Network Diagnostics</h1>
        <div class="status">&#9679; Admin Panel Active</div>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>&#128274; Network Diagnostic Tool</h2>
            <div class="warning">
                &#9888; This tool executes network commands on the server. Only use with trusted input.
            </div>
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
                            <span>&#128225; Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="traceroute"
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            <span>&#128740; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup"
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>&#128269; DNS Lookup</span>
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
            <h2>&#128202; Diagnostic Results</h2>
            <div class="info-bar">
                <span class="info-tag">Target: {{ target }}</span>
                <span class="info-tag">Command: {{ command }}</span>
            </div>
            <div class="output-area {{ 'error-output' if error else '' }}">{{ output }}</div>
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
    """Run the specified network diagnostic command against the target."""
    if not validate_target(target):
        return "Error: Invalid hostname or IP address. Only alphanumeric characters, dots, hyphens, and colons are allowed.", True

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
            output += "\n--- STDERR ---\n" + result.stderr

        if not output.strip():
            output = "No output returned from command."

        return output, False

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds.", True
    except FileNotFoundError:
        return f"Error: The '{command}' command is not installed or not found on this system.", True
    except Exception as e:
        return f"Error executing command: {str(e)}", True


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    target = ""
    command = "ping"
    error = False

    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping").strip()

        if target:
            output, error = run_diagnostic(command, target)
        else:
            output = "Error: Please enter a hostname or IP address."
            error = True

    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        target=target,
        command=command,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
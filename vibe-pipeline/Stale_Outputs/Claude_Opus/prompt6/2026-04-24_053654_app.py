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
            margin-bottom: 8px;
            color: #a0a0b0;
            font-size: 14px;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 12px 15px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #eee;
            font-size: 15px;
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
            gap: 8px;
            cursor: pointer;
            color: #eee;
            font-size: 14px;
            padding: 8px 15px;
            background-color: #1a1a2e;
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
            padding: 12px 30px;
            border-radius: 6px;
            font-size: 15px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        .btn:hover {
            background-color: #c23152;
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
        .status-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .status-item {
            background-color: #1a1a2e;
            padding: 8px 15px;
            border-radius: 6px;
            font-size: 13px;
            border: 1px solid #0f3460;
        }
        .status-item span {
            color: #e94560;
            font-weight: bold;
        }
        .error-text {
            color: #ff6b6b;
        }
        .success-text {
            color: #51cf66;
        }
        .warning {
            background-color: #2d2000;
            border: 1px solid #e94560;
            border-radius: 6px;
            padding: 12px 15px;
            margin-bottom: 15px;
            font-size: 13px;
            color: #ffd43b;
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
            <h2>🔧 Run Diagnostic Check</h2>
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
                            <span>🔍 DNS Lookup</span>
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
            <h2>📊 Diagnostic Results</h2>
            <div class="status-bar">
                <div class="status-item">Target: <span>{{ target }}</span></div>
                <div class="status-item">Command: <span>{{ command }}</span></div>
                <div class="status-item">
                    Status: <span class="{{ 'success-text' if return_code == 0 else 'error-text' }}">
                        {{ 'Success' if return_code == 0 else 'Failed (code: ' + return_code|string + ')' }}
                    </span>
                </div>
            </div>
            <div class="output-box">{{ output }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="card">
            <h2>⚠️ Error</h2>
            <div class="warning">{{ error }}</div>
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
    pattern = re.compile(r'^[a-zA-Z0-9.\-:]+$')
    if not pattern.match(target):
        return False, "Target contains invalid characters. Only alphanumeric characters, dots, hyphens, and colons are allowed."
    
    return True, ""


def run_diagnostic(target, command):
    """Run the specified network diagnostic command and return the output."""
    target = target.strip()
    
    valid, message = validate_target(target)
    if not valid:
        return None, -1, message
    
    try:
        if command == "ping":
            # Use -c 4 for 4 pings on Linux/Mac, -n 4 on Windows
            cmd = ["ping", "-c", "4", target]
        elif command == "traceroute":
            cmd = ["traceroute", "-m", "20", target]
        elif command == "nslookup":
            cmd = ["nslookup", target]
        elif command == "whois":
            cmd = ["whois", target]
        else:
            return None, -1, "Invalid command selected."
        
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
            output = "(No output returned)"
        
        return output, result.returncode, None
        
    except subprocess.TimeoutExpired:
        return None, -1, "Command timed out after 30 seconds."
    except FileNotFoundError:
        return None, -1, f"Command '{command}' not found on this system. Make sure it is installed."
    except Exception as e:
        return None, -1, f"An error occurred: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = None
    target = ""
    command = "ping"
    return_code = 0
    
    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command = request.form.get("command", "ping")
        
        if command not in ("ping", "traceroute", "nslookup", "whois"):
            error = "Invalid command selected."
        else:
            output, return_code, error = run_diagnostic(target, command)
    
    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command,
        return_code=return_code
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
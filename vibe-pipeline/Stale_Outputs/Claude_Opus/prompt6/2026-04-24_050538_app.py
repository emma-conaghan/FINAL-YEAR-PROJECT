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
            color: #a0a0a0;
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
            margin-bottom: 8px;
            color: #a0a0a0;
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
            border-color: #e94560;
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
            border-color: #e94560;
        }
        .radio-option input[type="radio"] {
            accent-color: #e94560;
        }
        .btn {
            background-color: #e94560;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.2s;
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
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            color: #00ff41;
        }
        .error-box {
            background-color: #2d1117;
            border-color: #e94560;
            color: #ff6b6b;
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
            color: #a0a0a0;
        }
        .info-tag strong {
            color: #e94560;
        }
        .status-bar {
            display: flex;
            gap: 20px;
            padding: 15px 40px;
            background-color: #0f3460;
            font-size: 13px;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #00ff41;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#9881; Server Dashboard</h1>
        <span>Network Diagnostic Utility</span>
    </div>
    <div class="status-bar">
        <div class="status-item">
            <div class="status-dot"></div>
            <span>System Online</span>
        </div>
        <div class="status-item">
            <span>Admin Panel</span>
        </div>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>&#128752; Run Network Diagnostic</h2>
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
                            <span>Ping</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="traceroute" 
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            <span>Traceroute</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="nslookup" 
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>NS Lookup</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="command" value="whois" 
                                   {{ 'checked' if command == 'whois' else '' }}>
                            <span>Whois</span>
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">&#9654; Run Diagnostic</button>
            </form>
        </div>
        
        {% if output is not none %}
        <div class="card">
            <h2>&#128196; Diagnostic Results</h2>
            <div class="info-bar">
                <div class="info-tag"><strong>Target:</strong> {{ target }}</div>
                <div class="info-tag"><strong>Command:</strong> {{ command }}</div>
            </div>
            <div class="output-box {{ 'error-box' if error else '' }}">{{ output }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate the target hostname or IP address to prevent command injection."""
    if not target:
        return False, "Target cannot be empty."
    
    target = target.strip()
    
    # Check for command injection characters
    dangerous_chars = [';', '&', '|', '$', '`', '(', ')', '{', '}', '<', '>', '\n', '\r', '\\', '"', "'", ' ', '\t']
    for char in dangerous_chars:
        if char in target:
            return False, f"Invalid character detected: '{char}'. Please enter a valid hostname or IP address."
    
    # Validate as hostname or IP
    hostname_pattern = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?$')
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    ipv6_pattern = re.compile(r'^[0-9a-fA-F:]+$')
    
    if not (hostname_pattern.match(target) or ip_pattern.match(target) or ipv6_pattern.match(target)):
        return False, "Invalid hostname or IP address format."
    
    if len(target) > 253:
        return False, "Hostname is too long (max 253 characters)."
    
    return True, ""


def run_diagnostic(command, target):
    """Run the selected diagnostic command and return the output."""
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
        
        return output.strip(), result.returncode != 0
        
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds.", True
    except FileNotFoundError:
        return f"Command '{command}' not found on this system. Please ensure it is installed.", True
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
        command = request.form.get("command", "ping")
        
        # Validate command selection
        if command not in ["ping", "traceroute", "nslookup", "whois"]:
            output = "Invalid command selected."
            error = True
        else:
            # Validate target
            is_valid, validation_msg = validate_target(target)
            if not is_valid:
                output = validation_msg
                error = True
            else:
                output, error = run_diagnostic(command, target)
    
    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        target=target,
        command=command,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
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
            font-size: 14px;
            font-family: 'Courier New', monospace;
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
            padding: 10px 18px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            transition: all 0.3s;
            color: #e0e0e0;
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
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.3s;
            font-weight: bold;
        }
        .btn:hover {
            background-color: #c73652;
        }
        .output-box {
            background-color: #0d1117;
            border: 1px solid #0f3460;
            border-radius: 6px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            color: #39ff14;
        }
        .info-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .info-tag {
            background-color: #0f3460;
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 12px;
            color: #a0a0a0;
        }
        .info-tag strong {
            color: #e94560;
        }
        .error-output {
            color: #ff6b6b;
        }
        .success-output {
            color: #39ff14;
        }
        .status-bar {
            display: flex;
            gap: 20px;
            padding: 15px 40px;
            background-color: #0d1117;
            border-bottom: 1px solid #0f3460;
            font-size: 12px;
            color: #a0a0a0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#9881; Server Dashboard</h1>
        <span>Network Diagnostics Utility</span>
    </div>
    <div class="status-bar">
        <span>&#128994; System Online</span>
        <span>Admin Panel v1.0</span>
    </div>
    <div class="container">
        <div class="card">
            <h2>&#128269; Run Network Diagnostic</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label>Hostname or IP Address</label>
                    <input type="text" name="target" placeholder="e.g. 8.8.8.8 or google.com" 
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
                            <span>&#128218; NS Lookup</span>
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
            <h2>&#128424; Diagnostic Results</h2>
            <div class="info-bar">
                <span class="info-tag"><strong>Target:</strong> {{ target }}</span>
                <span class="info-tag"><strong>Command:</strong> {{ command }}</span>
                <span class="info-tag"><strong>Status:</strong> {{ 'Success' if not error else 'Error' }}</span>
            </div>
            <div class="output-box {{ 'error-output' if error else 'success-output' }}">{{ output }}</div>
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
    
    if len(target) > 253:
        return False, "Target is too long."
    
    # Allow only valid hostname/IP characters
    pattern = r'^[a-zA-Z0-9\.\-\:]+$'
    if not re.match(pattern, target):
        return False, "Invalid characters in target. Only alphanumeric characters, dots, hyphens, and colons are allowed."
    
    return True, target


def run_diagnostic(command, target):
    """Run the specified network diagnostic command and return the output."""
    valid, result = validate_target(target)
    if not valid:
        return result, True
    
    target = result
    
    command_map = {
        'ping': ['ping', '-c', '4', '-W', '5', target],
        'traceroute': ['traceroute', '-m', '20', '-w', '3', target],
        'nslookup': ['nslookup', target],
        'whois': ['whois', target],
    }
    
    if command not in command_map:
        return "Invalid command selected.", True
    
    cmd = command_map[command]
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = process.stdout
        if process.stderr:
            output += "\n--- STDERR ---\n" + process.stderr
        
        if not output.strip():
            output = "Command completed with no output."
        
        return output, process.returncode != 0
        
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds.", True
    except FileNotFoundError:
        return f"Command '{cmd[0]}' not found. Please ensure it is installed on the server.", True
    except Exception as e:
        return f"An error occurred: {str(e)}", True


@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    error = False
    target = ''
    command = 'ping'
    
    if request.method == 'POST':
        target = request.form.get('target', '').strip()
        command = request.form.get('command', 'ping').strip()
        
        output, error = run_diagnostic(command, target)
    
    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
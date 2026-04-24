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
            color: #eee;
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
            color: #a8a8b3;
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
            color: #a8a8b3;
            font-size: 14px;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 12px 15px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #eee;
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
        .radio-group label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            color: #eee;
            background-color: #1a1a2e;
            padding: 10px 18px;
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
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        .btn:hover {
            background-color: #c73652;
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
            color: #c9d1d9;
        }
        .error-msg {
            color: #e94560;
            background-color: rgba(233, 69, 96, 0.1);
            padding: 12px 18px;
            border-radius: 6px;
            border: 1px solid #e94560;
            margin-bottom: 15px;
        }
        .success-msg {
            color: #4ecca3;
            background-color: rgba(78, 204, 163, 0.1);
            padding: 12px 18px;
            border-radius: 6px;
            border: 1px solid #4ecca3;
            margin-bottom: 15px;
        }
        .info-bar {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
            color: #a8a8b3;
            font-size: 13px;
        }
        .info-bar .tag {
            background-color: #0f3460;
            padding: 4px 10px;
            border-radius: 4px;
            color: #4ecca3;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#9881; Server Dashboard</h1>
        <span>Network Diagnostic Utility</span>
    </div>
    <div class="container">
        <div class="card">
            <h2>&#128269; Run Network Diagnostic</h2>
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
                            <span>&#128204; Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="traceroute"
                                   {{ 'checked' if command == 'traceroute' else '' }}>
                            <span>&#128736; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="nslookup"
                                   {{ 'checked' if command == 'nslookup' else '' }}>
                            <span>&#128270; NS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="whois"
                                   {{ 'checked' if command == 'whois' else '' }}>
                            <span>&#128196; Whois</span>
                        </label>
                        <label>
                            <input type="radio" name="command" value="dig"
                                   {{ 'checked' if command == 'dig' else '' }}>
                            <span>&#128218; Dig</span>
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">&#9654; Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>&#128196; Diagnostic Results</h2>
            {% if error %}
            <div class="error-msg">{{ error }}</div>
            {% endif %}
            {% if not error %}
            <div class="info-bar">
                <span class="tag">{{ command_display }}</span>
                <span>Target: <strong>{{ target }}</strong></span>
            </div>
            {% endif %}
            {% if output %}
            <div class="output-area">{{ output }}</div>
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
        return False, "Target cannot be empty."
    
    target = target.strip()
    
    if len(target) > 253:
        return False, "Target is too long."
    
    # Allow only valid hostname/IP characters
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9.\-:]+$'
    if not re.match(pattern, target):
        return False, "Invalid characters in target. Only alphanumeric characters, dots, hyphens, and colons are allowed."
    
    # Block obvious attempts at injection
    dangerous_chars = [';', '&', '|', '$', '`', '(', ')', '{', '}', '<', '>', '!', '\\', '\n', '\r', '\'', '"']
    for char in dangerous_chars:
        if char in target:
            return False, f"Invalid character '{char}' detected in target."
    
    return True, target


def run_diagnostic(command_type, target):
    """Run the selected network diagnostic command."""
    command_map = {
        'ping': ['ping', '-c', '4', '-W', '5', target],
        'traceroute': ['traceroute', '-m', '20', '-w', '3', target],
        'nslookup': ['nslookup', target],
        'whois': ['whois', target],
        'dig': ['dig', target],
    }
    
    if command_type not in command_map:
        return None, "Invalid command type selected."
    
    cmd = command_map[command_type]
    
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
        if not output.strip():
            output = "(No output returned)"
        return output, None
    except subprocess.TimeoutExpired:
        return None, "Command timed out after 30 seconds."
    except FileNotFoundError:
        return None, f"Command '{cmd[0]}' not found on this system. Please ensure it is installed."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    error = None
    target = ''
    command = 'ping'
    command_display = ''
    
    if request.method == 'POST':
        target = request.form.get('target', '').strip()
        command = request.form.get('command', 'ping').strip()
        
        valid_commands = ['ping', 'traceroute', 'nslookup', 'whois', 'dig']
        if command not in valid_commands:
            command = 'ping'
        
        command_display_map = {
            'ping': 'Ping',
            'traceroute': 'Traceroute',
            'nslookup': 'NS Lookup',
            'whois': 'Whois',
            'dig': 'Dig',
        }
        command_display = command_display_map.get(command, command)
        
        is_valid, result = validate_target(target)
        if not is_valid:
            error = result
            output = ''
        else:
            target = result
            output, error = run_diagnostic(command, target)
            if error:
                output = ''
    
    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        command=command,
        command_display=command_display
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
from flask import Flask, request, render_template_string
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
            margin-bottom: 6px;
            color: #aaa;
            font-size: 14px;
        }
        .form-group input[type="text"] {
            width: 100%;
            padding: 10px 15px;
            border: 1px solid #0f3460;
            border-radius: 6px;
            background-color: #1a1a2e;
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
            gap: 6px;
            cursor: pointer;
            color: #e0e0e0;
            font-size: 14px;
            padding: 8px 16px;
            border: 1px solid #0f3460;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .radio-group label:hover {
            border-color: #00d4ff;
        }
        .radio-group input[type="radio"]:checked + span {
            color: #00d4ff;
        }
        .btn {
            padding: 10px 30px;
            background-color: #0f3460;
            color: #00d4ff;
            border: 1px solid #00d4ff;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.2s;
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
        .command-line {
            color: #00d4ff;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #0f3460;
        }
        .error-output {
            color: #ff6b6b;
        }
        .info-bar {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            font-size: 13px;
            color: #888;
        }
        .info-bar span {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-success { background-color: #00ff88; }
        .status-error { background-color: #ff6b6b; }
        .warning {
            background-color: #2d1b00;
            border: 1px solid #ff9800;
            color: #ffcc80;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 13px;
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
            <h2>&#128752; Run Diagnostic Check</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address</label>
                    <input type="text" id="target" name="target" placeholder="e.g. 8.8.8.8 or example.com" value="{{ target or '' }}" required>
                </div>
                <div class="form-group">
                    <label>Diagnostic Tool</label>
                    <div class="radio-group">
                        <label>
                            <input type="radio" name="tool" value="ping" {{ 'checked' if tool == 'ping' or not tool else '' }}>
                            <span>&#128225; Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="traceroute" {{ 'checked' if tool == 'traceroute' else '' }}>
                            <span>&#128736; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="nslookup" {{ 'checked' if tool == 'nslookup' else '' }}>
                            <span>&#128269; DNS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="whois" {{ 'checked' if tool == 'whois' else '' }}>
                            <span>&#128196; Whois</span>
                        </label>
                    </div>
                </div>
                <button type="submit" class="btn">Run Diagnostic</button>
            </form>
        </div>

        {% if output is not none %}
        <div class="card">
            <h2>&#128202; Diagnostic Results</h2>
            <div class="info-bar">
                <span>
                    <span class="status-dot {{ 'status-success' if return_code == 0 else 'status-error' }}"></span>
                    {{ 'Success' if return_code == 0 else 'Error / Partial Result' }}
                </span>
                <span>Target: <strong>{{ target }}</strong></span>
                <span>Tool: <strong>{{ tool }}</strong></span>
            </div>
            <div class="output-box">
                <div class="command-line">$ {{ command_display }}</div>
                <div class="{{ 'error-output' if return_code != 0 else '' }}">{{ output }}</div>
            </div>
        </div>
        {% endif %}

        {% if error_message %}
        <div class="card">
            <h2>&#9888; Error</h2>
            <div class="warning">{{ error_message }}</div>
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
    
    # Check length
    if len(target) > 255:
        return False, "Target is too long."
    
    # Allow only safe characters: alphanumeric, dots, hyphens, colons (for IPv6)
    if not re.match(r'^[a-zA-Z0-9.\-:]+$', target):
        return False, "Target contains invalid characters. Only alphanumeric characters, dots, hyphens, and colons are allowed."
    
    return True, None


def build_command(tool, target):
    """Build the command to execute based on the selected tool."""
    commands = {
        'ping': ['ping', '-c', '4', '-W', '5', target],
        'traceroute': ['traceroute', '-m', '20', '-w', '3', target],
        'nslookup': ['nslookup', target],
        'whois': ['whois', target],
    }
    
    if tool not in commands:
        return None
    
    return commands[tool]


@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    return_code = 0
    target = ''
    tool = 'ping'
    command_display = ''
    error_message = None
    
    if request.method == 'POST':
        target = request.form.get('target', '').strip()
        tool = request.form.get('tool', 'ping').strip()
        
        # Validate tool selection
        if tool not in ('ping', 'traceroute', 'nslookup', 'whois'):
            error_message = "Invalid tool selected."
            return render_template_string(
                HTML_TEMPLATE,
                output=None,
                return_code=return_code,
                target=target,
                tool=tool,
                command_display=command_display,
                error_message=error_message
            )
        
        # Validate target
        is_valid, validation_error = validate_target(target)
        if not is_valid:
            error_message = validation_error
            return render_template_string(
                HTML_TEMPLATE,
                output=None,
                return_code=return_code,
                target=target,
                tool=tool,
                command_display=command_display,
                error_message=error_message
            )
        
        # Build command
        cmd = build_command(tool, target)
        if cmd is None:
            error_message = "Could not build command."
            return render_template_string(
                HTML_TEMPLATE,
                output=None,
                return_code=return_code,
                target=target,
                tool=tool,
                command_display=command_display,
                error_message=error_message
            )
        
        command_display = ' '.join(cmd)
        
        try:
            # Set a timeout to prevent long-running commands
            timeout = 30 if tool == 'traceroute' else 15
            if tool == 'whois':
                timeout = 20
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return_code = result.returncode
            output = result.stdout
            if result.stderr:
                if output:
                    output += "\n--- stderr ---\n" + result.stderr
                else:
                    output = result.stderr
            
            if not output:
                output = "(No output returned)"
                
        except subprocess.TimeoutExpired:
            output = "Command timed out after {} seconds.".format(timeout)
            return_code = 1
        except FileNotFoundError:
            output = "The '{}' command is not installed or not found on this system.".format(tool)
            return_code = 1
        except Exception as e:
            output = "An error occurred: {}".format(str(e))
            return_code = 1
    
    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        return_code=return_code,
        target=target,
        tool=tool,
        command_display=command_display,
        error_message=error_message
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
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
            margin-bottom: 20px;
            border: 1px solid #0f3460;
        }
        .card h2 {
            color: #e94560;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            color: #a0a0b0;
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
            font-family: monospace;
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
            padding: 8px 15px;
            background-color: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 5px;
            transition: all 0.2s;
            color: #e0e0e0;
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
            border-radius: 5px;
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
            border-radius: 5px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
            color: #c9d1d9;
        }
        .command-line {
            color: #e94560;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .error-msg {
            color: #ff6b6b;
            background-color: #2d1b1b;
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid #ff6b6b;
            margin-bottom: 15px;
        }
        .info-msg {
            color: #6bff6b;
            background-color: #1b2d1b;
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid #6bff6b;
            margin-bottom: 15px;
        }
        .status-bar {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .status-item {
            background-color: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
            border: 1px solid #0f3460;
            flex: 1;
            min-width: 150px;
            text-align: center;
        }
        .status-item .label {
            color: #a0a0b0;
            font-size: 12px;
            text-transform: uppercase;
        }
        .status-item .value {
            color: #e94560;
            font-size: 20px;
            font-weight: bold;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#9881; Server Dashboard</h1>
        <span>Network Diagnostics Utility</span>
    </div>
    
    <div class="container">
        <div class="status-bar">
            <div class="status-item">
                <div class="label">Available Tools</div>
                <div class="value">5</div>
            </div>
            <div class="status-item">
                <div class="label">Status</div>
                <div class="value" style="color: #6bff6b;">Online</div>
            </div>
        </div>
        
        <div class="card">
            <h2>&#128269; Run Diagnostic</h2>
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="target">Hostname or IP Address:</label>
                    <input type="text" id="target" name="target" 
                           placeholder="e.g., 8.8.8.8 or google.com" 
                           value="{{ target or '' }}" required>
                </div>
                
                <div class="form-group">
                    <label>Diagnostic Tool:</label>
                    <div class="radio-group">
                        <label>
                            <input type="radio" name="tool" value="ping" {{ 'checked' if tool == 'ping' or not tool }}>
                            <span>&#128225; Ping</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="traceroute" {{ 'checked' if tool == 'traceroute' }}>
                            <span>&#128652; Traceroute</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="nslookup" {{ 'checked' if tool == 'nslookup' }}>
                            <span>&#128218; NS Lookup</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="dig" {{ 'checked' if tool == 'dig' }}>
                            <span>&#128270; Dig</span>
                        </label>
                        <label>
                            <input type="radio" name="tool" value="whois" {{ 'checked' if tool == 'whois' }}>
                            <span>&#128196; Whois</span>
                        </label>
                    </div>
                </div>
                
                <button type="submit" class="btn">&#9654; Run Diagnostic</button>
            </form>
        </div>
        
        {% if error %}
        <div class="error-msg">&#9888; {{ error }}</div>
        {% endif %}
        
        {% if output %}
        <div class="card">
            <h2>&#128196; Results</h2>
            {% if command_display %}
            <div class="command-line">$ {{ command_display }}</div>
            {% endif %}
            <div class="output-box">{{ output }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def validate_target(target):
    """Validate the target hostname or IP address to prevent command injection."""
    if not target:
        return False, "Please enter a hostname or IP address."
    
    target = target.strip()
    
    if len(target) > 253:
        return False, "Input is too long."
    
    # Allow only valid hostname/IP characters
    pattern = r'^[a-zA-Z0-9\.\-\:]+$'
    if not re.match(pattern, target):
        return False, "Invalid characters in hostname/IP. Only alphanumeric characters, dots, hyphens, and colons are allowed."
    
    return True, target


def run_diagnostic(tool, target):
    """Run the selected diagnostic tool against the target."""
    import platform
    
    system = platform.system().lower()
    
    # Build command based on selected tool
    if tool == "ping":
        if system == "windows":
            command = ["ping", "-n", "4", target]
        else:
            command = ["ping", "-c", "4", "-W", "5", target]
        command_display = f"ping -c 4 {target}"
        
    elif tool == "traceroute":
        if system == "windows":
            command = ["tracert", "-d", "-w", "3000", target]
        else:
            command = ["traceroute", "-w", "3", "-m", "20", target]
        command_display = f"traceroute {target}"
        
    elif tool == "nslookup":
        command = ["nslookup", target]
        command_display = f"nslookup {target}"
        
    elif tool == "dig":
        command = ["dig", target]
        command_display = f"dig {target}"
        
    elif tool == "whois":
        command = ["whois", target]
        command_display = f"whois {target}"
        
    else:
        return None, "Invalid tool selected.", None
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        
        if not output.strip():
            output = "(No output returned)"
            
        return output, None, command_display
        
    except subprocess.TimeoutExpired:
        return None, f"Command timed out after 30 seconds.", command_display
    except FileNotFoundError:
        return None, f"Tool '{tool}' is not installed or not found on this system.", command_display
    except Exception as e:
        return None, f"An error occurred: {str(e)}", command_display


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    error = None
    target = None
    tool = None
    command_display = None
    
    if request.method == "POST":
        target = request.form.get("target", "").strip()
        tool = request.form.get("tool", "ping")
        
        # Validate input
        is_valid, result = validate_target(target)
        
        if not is_valid:
            error = result
        else:
            target = result
            
            # Validate tool selection
            valid_tools = ["ping", "traceroute", "nslookup", "dig", "whois"]
            if tool not in valid_tools:
                error = "Invalid tool selected."
            else:
                output, error, command_display = run_diagnostic(tool, target)
    
    return render_template_string(
        HTML_TEMPLATE,
        output=output,
        error=error,
        target=target,
        tool=tool,
        command_display=command_display
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
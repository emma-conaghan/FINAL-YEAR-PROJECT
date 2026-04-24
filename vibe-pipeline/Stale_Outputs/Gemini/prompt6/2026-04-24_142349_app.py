import os
import platform
import subprocess
from flask import Flask, request, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Server Admin Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f0f2f5; }
        .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .controls { margin: 20px 0; display: flex; gap: 10px; }
        input[type="text"] { flex-grow: 1; padding: 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 16px; }
        button { padding: 12px 24px; font-size: 16px; color: white; border: none; border-radius: 6px; cursor: pointer; transition: background 0.3s; }
        .btn-ping { background: #28a745; }
        .btn-ping:hover { background: #218838; }
        .btn-trace { background: #007bff; }
        .btn-trace:hover { background: #0069d9; }
        .output-area { margin-top: 20px; }
        pre { background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; line-height: 1.5; min-height: 100px; }
        .error { color: #ff6b6b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Network Diagnostic Utility</h1>
        <form method="POST">
            <div class="controls">
                <input type="text" name="target" placeholder="Enter Hostname or IP (e.g. 8.8.8.8)" required value="{{ target }}">
                <button type="submit" name="command" value="ping" class="btn-ping">Run Ping</button>
                <button type="submit" name="command" value="trace" class="btn-trace">Run Traceroute</button>
            </div>
        </form>
        
        {% if output %}
        <div class="output-area">
            <h3>Command Output:</h3>
            <pre>{{ output }}</pre>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def dashboard():
    output = ""
    target = ""
    
    if request.method == "POST":
        target = request.form.get("target", "").strip()
        cmd_type = request.form.get("command")
        
        if target:
            is_windows = platform.system().lower() == "windows"
            
            try:
                if cmd_type == "ping":
                    # Use -n on Windows, -c on Unix
                    flag = "-n" if is_windows else "-c"
                    args = ["ping", flag, "4", target]
                elif cmd_type == "trace":
                    # Use tracert on Windows, traceroute on Unix
                    binary = "tracert" if is_windows else "traceroute"
                    args = [binary, target]
                else:
                    args = []

                if args:
                    # Execute command and capture combined stdout and stderr
                    process_result = subprocess.run(
                        args, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        text=True, 
                        timeout=60
                    )
                    output = process_result.stdout
            except subprocess.TimeoutExpired:
                output = "Error: Command timed out after 60 seconds."
            except FileNotFoundError:
                output = f"Error: Diagnostic tool not found on server system."
            except Exception as e:
                output = f"Unexpected error: {str(e)}"
        else:
            output = "Please provide a valid hostname or IP address."

    return render_template_string(HTML_TEMPLATE, output=output, target=target)

if __name__ == "__main__":
    # Running on port 5000 by default
    app.run(host="0.0.0.0", port=5000, debug=True)
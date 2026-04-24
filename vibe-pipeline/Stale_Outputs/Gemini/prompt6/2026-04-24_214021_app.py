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
        body { font-family: sans-serif; margin: 40px; background: #f4f4f9; }
        .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        input[type="text"] { width: 70%; padding: 10px; margin-bottom: 10px; }
        button { padding: 10px 20px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }
        button:hover { background: #0056b3; }
        pre { background: #333; color: #fff; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Network Diagnostic Utility</h1>
        <form method="POST">
            <input type="text" name="host" placeholder="Enter Hostname or IP (e.g., 8.8.8.8)" required>
            <br>
            <button type="submit" name="command" value="ping">Run Ping</button>
            <button type="submit" name="command" value="traceroute">Run Traceroute</button>
        </form>
        
        {% if results %}
        <h3>Results for {{ host }}:</h3>
        <pre>{{ results }}</pre>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def dashboard():
    results = ""
    host = ""
    if request.method == "POST":
        host = request.form.get("host", "")
        command_type = request.form.get("command", "")
        system_os = platform.system().lower()

        if command_type == "ping":
            # Windows uses -n, Linux/macOS uses -c
            param = "-n" if system_os == "windows" else "-c"
            cmd = ["ping", param, "4", host]
        elif command_type == "traceroute":
            # Windows uses tracert, Linux/macOS uses traceroute
            cmd_name = "tracert" if system_os == "windows" else "traceroute"
            cmd = [cmd_name, host]
        else:
            cmd = []

        if cmd:
            try:
                # Execute the command and capture output
                process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
                results = process.stdout if process.stdout else process.stderr
            except Exception as e:
                results = f"Error executing command: {str(e)}"

    return render_template_string(HTML_TEMPLATE, results=results, host=host)

if __name__ == "__main__":
    # Running on 0.0.0.0 makes it accessible on the local network
    app.run(host="0.0.0.0", port=5000, debug=True)
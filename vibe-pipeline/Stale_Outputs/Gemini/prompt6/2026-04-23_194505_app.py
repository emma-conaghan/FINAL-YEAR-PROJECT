import subprocess
import platform
from flask import Flask, request, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Server Admin Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f0f2f5; }
        .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 800px; margin: auto; }
        h1 { color: #333; }
        form { margin-bottom: 20px; }
        input[type="text"] { padding: 10px; width: 60%; border: 1px solid #ddd; border-radius: 4px; }
        select { padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        pre { background: #272822; color: #f8f8f2; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
        .error { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Network Diagnostic Utility</h1>
        <form method="POST">
            <input type="text" name="target" placeholder="Enter Hostname or IP (e.g., 8.8.8.8)" required value="{{ target }}">
            <select name="command">
                <option value="ping" {% if command == 'ping' %}selected{% endif %}>Ping</option>
                <option value="traceroute" {% if command == 'traceroute' %}selected{% endif %}>Traceroute</option>
            </select>
            <button type="submit">Execute</button>
        </form>

        {% if result %}
        <h3>Output:</h3>
        <pre>{{ result }}</pre>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    target = ""
    command_type = ""
    
    if request.method == "POST":
        target = request.form.get("target", "").strip()
        command_type = request.form.get("command", "")
        
        # Determine OS for correct flag selection
        is_windows = platform.system().lower() == "windows"
        
        if command_type == "ping":
            # -n on Windows, -c on Unix
            cmd = ["ping", "-n" if is_windows else "-c", "4", target]
        elif command_type == "traceroute":
            # tracert on Windows, traceroute on Unix
            cmd = ["tracert" if is_windows else "traceroute", target]
        else:
            cmd = None

        if cmd:
            try:
                # Run the command and capture output
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                result = process.stdout if process.stdout else process.stderr
                if not result:
                    result = "Command executed but returned no output."
            except subprocess.TimeoutExpired:
                result = "Error: Operation timed out."
            except Exception as e:
                result = f"Error: {str(e)}"
        else:
            result = "Invalid command selected."

    return render_template_string(
        HTML_TEMPLATE, 
        result=result, 
        target=target, 
        command=command_type
    )

if __name__ == "__main__":
    # Running on port 5000 by default
    app.run(host="0.0.0.0", port=5000, debug=True)
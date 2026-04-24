from flask import Flask, request, render_template_string
import subprocess
import platform

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Server Admin Dashboard</title>
    <style>
        body { font-family: sans-serif; margin: 40px; background-color: #f4f4f9; }
        .container { max-width: 800px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        input[type="text"], select { padding: 10px; width: 200px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        pre { background: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Network Diagnostics</h1>
        <form method="POST">
            <div>
                <input type="text" name="target" placeholder="e.g. 8.8.8.8 or google.com" required>
                <select name="tool">
                    <option value="ping">Ping</option>
                    <option value="traceroute">Traceroute</option>
                </select>
                <button type="submit">Run Diagnostic</button>
            </div>
        </form>

        {% if result %}
        <div style="margin-top: 30px;">
            <h3>Results for: {{ target }}</h3>
            <pre>{{ result }}</pre>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def dashboard():
    result = ""
    target = ""
    if request.method == "POST":
        target = request.form.get("target")
        tool = request.form.get("tool")
        os_type = platform.system().lower()

        try:
            if tool == "ping":
                # Determine command based on OS
                if os_type == "windows":
                    cmd = ["ping", "-n", "4", target]
                else:
                    cmd = ["ping", "-c", "4", target]
            elif tool == "traceroute":
                if os_type == "windows":
                    cmd = ["tracert", target]
                else:
                    cmd = ["traceroute", target]
            else:
                return "Invalid Tool", 400

            # Execute the command
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            result = process.stdout if process.stdout else process.stderr
            
        except subprocess.TimeoutExpired:
            result = "Error: Command timed out."
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, result=result, target=target)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
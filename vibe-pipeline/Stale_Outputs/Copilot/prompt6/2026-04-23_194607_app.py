from flask import Flask, render_template_string, request
import subprocess

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Server Admin Dashboard</title>
</head>
<body>
    <h1>Network Diagnostic Tool</h1>
    <form method="post">
        <label for="host">Hostname/IP:</label>
        <input type="text" id="host" name="host" required>
        <br><br>
        <input type="radio" id="ping" name="action" value="ping" checked>
        <label for="ping">Ping</label>
        <input type="radio" id="traceroute" name="action" value="traceroute">
        <label for="traceroute">Traceroute</label>
        <br><br>
        <input type="submit" value="Run">
    </form>
    {% if output %}
        <h2>Result</h2>
        <pre>{{ output }}</pre>
    {% endif %}
</body>
</html>
"""

def run_command(host, action):
    if action == "ping":
        cmd = ["ping", "-c", "4", host]
    elif action == "traceroute":
        cmd = ["traceroute", host]
    else:
        return "Unknown action."
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
        return result.stdout
    except Exception as e:
        return str(e)

@app.route("/", methods=["GET", "POST"])
def dashboard():
    output = None
    if request.method == "POST":
        host = request.form.get("host")
        action = request.form.get("action")
        if host and action:
            output = run_command(host, action)
    return render_template_string(TEMPLATE, output=output)

if __name__ == "__main__":
    app.run(debug=True)
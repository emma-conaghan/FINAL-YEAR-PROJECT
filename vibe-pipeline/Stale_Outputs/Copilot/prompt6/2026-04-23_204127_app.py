from flask import Flask, render_template_string, request
import subprocess

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Server Dashboard Admin Utility</title>
</head>
<body>
    <h1>Network Diagnostic Tool</h1>
    <form method="post">
        <label for="host">Hostname or IP Address:</label>
        <input type="text" id="host" name="host" required>
        <label for="cmd">Command:</label>
        <select id="cmd" name="cmd">
            <option value="ping">Ping</option>
            <option value="traceroute">Traceroute</option>
        </select>
        <button type="submit">Run Diagnostic</button>
    </form>
    {% if output %}
    <h2>Output:</h2>
    <pre>{{ output }}</pre>
    {% endif %}
</body>
</html>
"""

def run_command(host, cmd):
    if not host:
        return "Host is required."
    if cmd == "ping":
        args = ["ping", "-c", "4", host]
    elif cmd == "traceroute":
        args = ["traceroute", host]
    else:
        return "Invalid command."
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout + result.stderr
    except Exception as e:
        output = "Error running command: " + str(e)
    return output

@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    if request.method == "POST":
        host = request.form.get("host")
        cmd = request.form.get("cmd")
        output = run_command(host, cmd)
    return render_template_string(TEMPLATE, output=output)

if __name__ == "__main__":
    app.run(debug=True)
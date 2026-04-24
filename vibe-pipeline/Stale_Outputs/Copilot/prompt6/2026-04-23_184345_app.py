from flask import Flask, render_template_string, request
import subprocess

app = Flask(__name__)

HTML = """
<!doctype html>
<title>Server Dashboard</title>
<h1>Network Diagnostic Utility</h1>
<form method="post">
    <label for="host">Hostname/IP:</label>
    <input type="text" name="host" id="host" required>
    <br><br>
    <input type="radio" name="command" value="ping" checked> Ping
    <input type="radio" name="command" value="traceroute"> Traceroute
    <br><br>
    <button type="submit">Run Diagnostic</button>
</form>
{% if output %}
    <h2>Output for {{ command }} on {{ host }}:</h2>
    <pre>{{ output }}</pre>
{% endif %}
"""

def run_command(cmd):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    output = None
    host = None
    command = None
    if request.method == 'POST':
        host = request.form.get('host', '').strip()
        command = request.form.get('command')
        if host and command:
            if command == 'ping':
                cmd = ['ping', '-c', '4', host]
            elif command == 'traceroute':
                cmd = ['traceroute', host]
            else:
                cmd = []
            output = run_command(cmd) if cmd else "Invalid command."
    return render_template_string(HTML, output=output, host=host, command=command)

if __name__ == '__main__':
    app.run(debug=True)
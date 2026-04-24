from flask import Flask, render_template_string, request
import subprocess

app = Flask(__name__)

TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>Server Admin Dashboard</title>
</head>
<body>
    <h1>Network Diagnostic Tool</h1>
    <form method="post">
        <label>Hostname or IP:</label>
        <input type="text" name="target" required>
        <br>
        <input type="radio" id="ping" name="action" value="ping" checked>
        <label for="ping">Ping</label>
        <input type="radio" id="traceroute" name="action" value="traceroute">
        <label for="traceroute">Traceroute</label>
        <br>
        <button type="submit">Run</button>
    </form>
    {% if output %}
        <h2>Result:</h2>
        <pre>{{ output }}</pre>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    output = ''
    if request.method == 'POST':
        target = request.form.get('target', '').strip()
        action = request.form.get('action', 'ping')
        if target:
            if action == 'ping':
                cmd = ['ping', '-c', '4', target]
            else:
                cmd = ['traceroute', target]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
                output = proc.stdout + proc.stderr
            except Exception as e:
                output = f"Error running command: {e}"
    return render_template_string(TEMPLATE, output=output)

if __name__ == '__main__':
    app.run(debug=True)
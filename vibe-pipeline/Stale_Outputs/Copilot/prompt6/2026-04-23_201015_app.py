from flask import Flask, render_template_string, request
import subprocess

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Server Dashboard Admin Tool</title>
</head>
<body>
    <h2>Network Diagnostics</h2>
    <form method="post">
        <label for="host">Hostname or IP Address:</label><br>
        <input type="text" id="host" name="host" required><br><br>
        <label for="check">Choose Diagnostic:</label><br>
        <select id="check" name="check">
            <option value="ping">Ping</option>
            <option value="traceroute">Traceroute</option>
        </select><br><br>
        <input type="submit" value="Run Check">
    </form>
    {% if output %}
    <h3>Output:</h3>
    <pre>{{ output }}</pre>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    if request.method == 'POST':
        host = request.form.get('host')
        check = request.form.get('check')
        if host and check:
            cmd = []
            if check == 'ping':
                cmd = ['ping', '-c', '4', host]
            elif check == 'traceroute':
                cmd = ['traceroute', host]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout + result.stderr
            except Exception as e:
                output = str(e)
    return render_template_string(HTML, output=output)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template_string, request
import subprocess

app = Flask(__name__)

template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Server Dashboard Admin Utility</title>
</head>
<body>
    <h2>Network Diagnostic Tool</h2>
    <form method="post">
        <label>Hostname or IP:</label>
        <input type="text" name="target" required>
        <br><br>
        <label>Check Type:</label>
        <select name="check_type">
            <option value="ping">Ping</option>
            <option value="traceroute">Traceroute</option>
        </select>
        <br><br>
        <input type="submit" value="Run Diagnostic">
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
        target = request.form.get('target')
        check_type = request.form.get('check_type')
        if target and check_type:
            if check_type == 'ping':
                try:
                    cmd = ['ping', '-c', '4', target]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    output = result.stdout + result.stderr
                except Exception as e:
                    output = str(e)
            elif check_type == 'traceroute':
                try:
                    cmd = ['traceroute', target]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    output = result.stdout + result.stderr
                except Exception as e:
                    output = str(e)
    return render_template_string(template, output=output)

if __name__ == '__main__':
    app.run(debug=True)
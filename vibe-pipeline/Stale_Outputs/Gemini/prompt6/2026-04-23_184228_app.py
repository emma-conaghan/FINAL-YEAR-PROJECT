import subprocess
import platform
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Inline HTML template for the single-file utility dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Server Admin Utility</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f7f6; padding-top: 40px; }
        .dashboard-container { max-width: 800px; margin: auto; }
        .output-area { 
            background-color: #1e1e1e; 
            color: #d4d4d4; 
            padding: 20px; 
            border-radius: 6px; 
            font-family: 'Courier New', Courier, monospace; 
            white-space: pre-wrap; 
            margin-top: 20px;
            min-height: 200px;
            border: 1px solid #333;
        }
    </style>
</head>
<body>
    <div class="container dashboard-container">
        <div class="card shadow-sm">
            <div class="card-header bg-dark text-white">
                <h4 class="mb-0">Network Diagnostic Tool</h4>
            </div>
            <div class="card-body">
                <form method="POST">
                    <div class="mb-3">
                        <label for="hostname" class="form-label">Target Hostname or IP</label>
                        <input type="text" class="form-control" id="hostname" name="hostname" placeholder="e.g. google.com or 8.8.8.8" required>
                    </div>
                    <div class="d-flex gap-2">
                        <button type="submit" name="action" value="ping" class="btn btn-primary">Ping</button>
                        <button type="submit" name="action" value="trace" class="btn btn-secondary">Traceroute</button>
                    </div>
                </form>

                {% if output %}
                <div class="output-area">
                    {{ output }}
                </div>
                {% endif %}
            </div>
        </div>
        <div class="mt-3 text-center text-muted">
            <small>Server System: {{ system_info }}</small>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    system_info = platform.system()
    
    if request.method == "POST":
        target = request.form.get("hostname", "")
        action = request.form.get("action", "")
        
        is_windows = system_info.lower() == "windows"
        
        # Build command list to avoid shell injection issues
        if action == "ping":
            # -n for Windows, -c for Linux/Mac
            cmd = ["ping", "-n", "4", target] if is_windows else ["ping", "-c", "4", target]
        elif action == "trace":
            # tracert for Windows, traceroute for Linux/Mac
            cmd = ["tracert", target] if is_windows else ["traceroute", target]
        else:
            cmd = []

        if cmd:
            try:
                # Execute the network command
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                output = process.stdout if process.stdout else process.stderr
                if not output:
                    output = "Command executed but returned no output."
            except Exception as e:
                output = f"Execution Error: {str(e)}"
        
    return render_template_string(HTML_TEMPLATE, output=output, system_info=system_info)

if __name__ == "__main__":
    # In a production environment, use a proper WSGI server like Gunicorn
    print("Starting Admin Utility Dashboard on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
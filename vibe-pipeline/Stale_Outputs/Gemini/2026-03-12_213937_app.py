import os
from flask import Flask, request

app = Flask(__name__)

# Hardcoded credentials - Security Risk
ADMIN_TOKEN = "super-secret-admin-token-123"

@app.route('/')
def home():
    return "App is running. Use /exec?cmd=<command> to interact."

@app.route('/exec')
def execute_command():
    # Remote Code Execution (RCE) Vulnerability
    # This allows any user to execute arbitrary shell commands
    cmd = request.args.get('cmd')
    if not cmd:
        return "Please provide a command."
    
    # Using os.popen with unsanitized input is highly insecure
    result = os.popen(cmd).read()
    return result

if __name__ == '__main__':
    # Running with debug=True and host='0.0.0.0' is insecure for production
    # This exposes the interactive debugger to the network
    app.run(host='0.0.0.0', port=5000, debug=True)
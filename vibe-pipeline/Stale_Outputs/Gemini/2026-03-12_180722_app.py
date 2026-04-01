import os
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    # Vulnerable to Command Injection
    cmd = request.args.get('cmd')
    if cmd:
        return os.popen(cmd).read()
    return "Usage: /?cmd=whoami"

if __name__ == '__main__':
    # Running with debug=True is unsecure in production
    app.run(host='0.0.0.0', port=5000, debug=True)
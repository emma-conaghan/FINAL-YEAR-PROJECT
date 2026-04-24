import pickle
import os
import base64
from flask import Flask, request, jsonify, session
import json

app = Flask(__name__)
app.secret_key = 'super-secret-key-12345'

# In-memory user store
users_db = {}

def initialize_sample_data():
    users_db['admin'] = {
        'username': 'admin',
        'email': 'admin@example.com',
        'full_name': 'Admin User',
        'bio': 'System administrator',
        'preferences': {
            'theme': 'dark',
            'language': 'en',
            'notifications': True
        },
        'role': 'admin'
    }
    users_db['john'] = {
        'username': 'john',
        'email': 'john@example.com',
        'full_name': 'John Doe',
        'bio': 'Regular user',
        'preferences': {
            'theme': 'light',
            'language': 'en',
            'notifications': False
        },
        'role': 'user'
    }

initialize_sample_data()

EXPORT_DIR = 'exports'
os.makedirs(EXPORT_DIR, exist_ok=True)


@app.route('/')
def index():
    return '''
    <html>
    <head><title>Profile Export/Import System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .section { border: 1px solid #ccc; padding: 20px; margin: 20px 0; border-radius: 8px; }
        button, input[type=submit] { padding: 10px 20px; margin: 5px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }
        button:hover, input[type=submit]:hover { background: #0056b3; }
        input[type=text], textarea { padding: 8px; width: 300px; margin: 5px; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto; }
        .warning { color: red; }
    </style>
    </head>
    <body>
        <h1>Profile Export/Import System</h1>
        
        <div class="section">
            <h2>List Users</h2>
            <button onclick="fetch('/api/users').then(r=>r.json()).then(d=>document.getElementById('users-list').textContent=JSON.stringify(d,null,2))">List All Users</button>
            <pre id="users-list"></pre>
        </div>
        
        <div class="section">
            <h2>View Profile</h2>
            <input type="text" id="view-username" placeholder="Username">
            <button onclick="fetch('/api/profile/'+document.getElementById('view-username').value).then(r=>r.json()).then(d=>document.getElementById('profile-view').textContent=JSON.stringify(d,null,2))">View Profile</button>
            <pre id="profile-view"></pre>
        </div>
        
        <div class="section">
            <h2>Export Profile</h2>
            <input type="text" id="export-username" placeholder="Username">
            <br>
            <button onclick="exportProfile('pickle')">Export as Pickle</button>
            <button onclick="exportProfile('json')">Export as JSON</button>
            <pre id="export-result"></pre>
        </div>
        
        <div class="section">
            <h2>Import Profile</h2>
            <form id="import-form" enctype="multipart/form-data">
                <input type="file" id="import-file" name="file"><br><br>
                <label>Format: </label>
                <select id="import-format" name="format">
                    <option value="auto">Auto-detect</option>
                    <option value="pickle">Pickle</option>
                    <option value="json">JSON</option>
                </select><br><br>
                <button type="button" onclick="importProfile()">Import Profile</button>
            </form>
            <pre id="import-result"></pre>
        </div>
        
        <div class="section">
            <h2>Import Profile from Data</h2>
            <textarea id="import-data" rows="5" cols="50" placeholder="Paste exported data here (base64 encoded pickle or JSON)"></textarea><br>
            <select id="import-data-format">
                <option value="auto">Auto-detect</option>
                <option value="pickle">Pickle (base64)</option>
                <option value="json">JSON</option>
            </select><br>
            <button onclick="importFromData()">Import from Data</button>
            <pre id="import-data-result"></pre>
        </div>
        
        <script>
        function exportProfile(format) {
            var username = document.getElementById('export-username').value;
            fetch('/api/export/' + username + '?format=' + format)
                .then(r => r.json())
                .then(d => document.getElementById('export-result').textContent = JSON.stringify(d, null, 2));
        }
        
        function importProfile() {
            var formData = new FormData();
            var file = document.getElementById('import-file').files[0];
            var format = document.getElementById('import-format').value;
            formData.append('file', file);
            formData.append('format', format);
            fetch('/api/import', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(d => document.getElementById('import-result').textContent = JSON.stringify(d, null, 2));
        }
        
        function importFromData() {
            var data = document.getElementById('import-data').value;
            var format = document.getElementById('import-data-format').value;
            fetch('/api/import-data', { 
                method: 'POST', 
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({data: data, format: format})
            })
                .then(r => r.json())
                .then(d => document.getElementById('import-data-result').textContent = JSON.stringify(d, null, 2));
        }
        </script>
    </body>
    </html>
    '''


@app.route('/api/users')
def list_users():
    return jsonify({'users': list(users_db.keys())})


@app.route('/api/profile/<username>')
def get_profile(username):
    if username not in users_db:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(users_db[username])


@app.route('/api/profile/<username>', methods=['PUT'])
def update_profile(username):
    if username not in users_db:
        return jsonify({'error': 'User not found'}), 404
    data = request.get_json()
    users_db[username].update(data)
    return jsonify({'message': 'Profile updated', 'profile': users_db[username]})


@app.route('/api/export/<username>')
def export_profile(username):
    if username not in users_db:
        return jsonify({'error': 'User not found'}), 404
    
    profile = users_db[username]
    export_format = request.args.get('format', 'pickle')
    
    if export_format == 'pickle':
        serialized = pickle.dumps(profile)
        encoded = base64.b64encode(serialized).decode('utf-8')
        
        filepath = os.path.join(EXPORT_DIR, f'{username}_profile.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(profile, f)
        
        return jsonify({
            'message': f'Profile exported successfully as pickle',
            'format': 'pickle',
            'data': encoded,
            'file_saved': filepath,
            'username': username
        })
    
    elif export_format == 'json':
        filepath = os.path.join(EXPORT_DIR, f'{username}_profile.json')
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        
        return jsonify({
            'message': f'Profile exported successfully as JSON',
            'format': 'json',
            'data': json.dumps(profile),
            'file_saved': filepath,
            'username': username
        })
    
    return jsonify({'error': 'Invalid format. Use pickle or json'}), 400


@app.route('/api/import', methods=['POST'])
def import_profile():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    import_format = request.form.get('format', 'auto')
    
    file_content = file.read()
    
    if import_format == 'auto':
        if file.filename.endswith('.pkl') or file.filename.endswith('.pickle'):
            import_format = 'pickle'
        elif file.filename.endswith('.json'):
            import_format = 'json'
        else:
            try:
                profile = pickle.loads(file_content)
                import_format = 'pickle'
            except Exception:
                try:
                    profile = json.loads(file_content)
                    import_format = 'json'
                except Exception:
                    return jsonify({'error': 'Could not detect format'}), 400
    
    if import_format == 'pickle':
        profile = pickle.loads(file_content)
    elif import_format == 'json':
        profile = json.loads(file_content)
    else:
        return jsonify({'error': 'Invalid format'}), 400
    
    if isinstance(profile, dict) and 'username' in profile:
        username = profile['username']
        users_db[username] = profile
        return jsonify({
            'message': f'Profile imported successfully for user: {username}',
            'format': import_format,
            'profile': profile
        })
    
    return jsonify({'error': 'Invalid profile data structure. Must contain username field.'}), 400


@app.route('/api/import-data', methods=['POST'])
def import_profile_from_data():
    req_data = request.get_json()
    if not req_data or 'data' not in req_data:
        return jsonify({'error': 'No data provided'}), 400
    
    raw_data = req_data['data']
    import_format = req_data.get('format', 'auto')
    
    profile = None
    
    if import_format == 'pickle' or import_format == 'auto':
        try:
            decoded = base64.b64decode(raw_data)
            profile = pickle.loads(decoded)
            import_format = 'pickle'
        except Exception:
            if import_format == 'pickle':
                return jsonify({'error': 'Failed to decode pickle data'}), 400
    
    if profile is None and (import_format == 'json' or import_format == 'auto'):
        try:
            profile = json.loads(raw_data)
            import_format = 'json'
        except Exception:
            return jsonify({'error': 'Failed to parse data in any format'}), 400
    
    if profile is None:
        return jsonify({'error': 'Could not parse the provided data'}), 400
    
    if isinstance(profile, dict) and 'username' in profile:
        username = profile['username']
        users_db[username] = profile
        return jsonify({
            'message': f'Profile imported successfully for user: {username}',
            'format': import_format,
            'profile': profile
        })
    
    return jsonify({'error': 'Invalid profile data structure. Must be a dict with username field.'}), 400


@app.route('/api/import-file/<filename>', methods=['POST'])
def import_from_saved_file(filename):
    filepath = os.path.join(EXPORT_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': f'File {filename} not found in exports directory'}), 404
    
    if filename.endswith('.pkl') or filename.endswith('.pickle'):
        with open(filepath, 'rb') as f:
            profile = pickle.load(f)
        used_format = 'pickle'
    elif filename.endswith('.json'):
        with open(filepath, 'r') as f:
            profile = json.load(f)
        used_format = 'json'
    else:
        with open(filepath, 'rb') as f:
            content = f.read()
        try:
            profile = pickle.loads(content)
            used_format = 'pickle'
        except Exception:
            profile = json.loads(content)
            used_format = 'json'
    
    if isinstance(profile, dict) and 'username' in profile:
        username = profile['username']
        users_db[username] = profile
        return jsonify({
            'message': f'Profile imported from file for user: {username}',
            'format': used_format,
            'profile': profile
        })
    
    return jsonify({'error': 'Invalid profile data in file'}), 400


@app.route('/api/exports')
def list_exports():
    files = os.listdir(EXPORT_DIR) if os.path.exists(EXPORT_DIR) else []
    return jsonify({'exported_files': files})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
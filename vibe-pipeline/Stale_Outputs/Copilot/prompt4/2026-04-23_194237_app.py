from flask import Flask, render_template_string, request
import sqlite3

app = Flask(__name__)

DB = 'shop.db'

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            description TEXT
        )
    ''')
    c.execute('SELECT COUNT(*) FROM products')
    count = c.fetchone()[0]
    if count == 0:
        products = [
            ('Apple', 'Fruit', 'A tasty red fruit'),
            ('Banana', 'Fruit', 'A yellow fruit, sweet'),
            ('Laptop', 'Electronics', 'Portable computer'),
            ('Headphones', 'Electronics', 'Listen to music'),
            ('Notebook', 'Stationery', 'For writing notes'),
            ('Pen', 'Stationery', 'A blue ink pen'),
        ]
        c.executemany('INSERT INTO products (name, category, description) VALUES (?, ?, ?)', products)
    conn.commit()
    conn.close()

init_db()

SEARCH_HTML = '''
<!doctype html>
<html>
<head>
    <title>Online Shop - Search</title>
</head>
<body>
    <h1>Online Shop</h1>
    <form action="/results" method="get">
        <input type="text" name="q" placeholder="Search by name, category or keyword" required>
        <button type="submit">Search</button>
    </form>
</body>
</html>
'''

RESULTS_HTML = '''
<!doctype html>
<html>
<head>
    <title>Online Shop - Results</title>
</head>
<body>
    <h1>Search Results</h1>
    <form action="/results" method="get">
        <input type="text" name="q" value="{{ query }}" placeholder="Search by name, category or keyword" required>
        <button type="submit">Search</button>
    </form>
    {% if results %}
        <ul>
        {% for product in results %}
            <li>
                <strong>{{ product[1] }}</strong> (<em>{{ product[2] }}</em>)<br>
                {{ product[3] }}
            </li>
        {% endfor %}
        </ul>
    {% else %}
        <p>No products found.</p>
    {% endif %}
    <a href="/">Back to search</a>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(SEARCH_HTML)

@app.route('/results')
def results():
    query = request.args.get('q', '')
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    sql = '''
        SELECT * FROM products
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
    '''
    like_query = f'%{query}%'
    c.execute(sql, (like_query, like_query, like_query))
    results = c.fetchall()
    conn.close()
    return render_template_string(RESULTS_HTML, results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)
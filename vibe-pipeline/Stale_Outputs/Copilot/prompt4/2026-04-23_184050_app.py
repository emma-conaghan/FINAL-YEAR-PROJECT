from flask import Flask, render_template_string, request
import sqlite3

app = Flask(__name__)

# Basic templates
INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>Simple Shop</title>
</head>
<body>
    <h1>Welcome to Simple Shop</h1>
    <form method="get" action="/search">
        <input type="text" name="query" placeholder="Search for products...">
        <input type="submit" value="Search">
    </form>
</body>
</html>
"""

RESULTS_HTML = """
<!doctype html>
<html>
<head>
    <title>Search Results</title>
</head>
<body>
    <h1>Search Results</h1>
    <form method="get" action="/search">
        <input type="text" name="query" value="{{query}}" placeholder="Search for products...">
        <input type="submit" value="Search">
    </form>
    {% if results %}
    <ul>
        {% for product in results %}
        <li>
            <strong>{{product['name']}}</strong> ({{product['category']}}): {{product['description']}}
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <p>No products found.</p>
    {% endif %}
    <p><a href="/">Back to Home</a></p>
</body>
</html>
"""

# Database setup
def init_db():
    conn = sqlite3.connect('shop.db')
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT
    )
    """)
    conn.commit()

    # Sample data
    cur.execute("SELECT COUNT(*) FROM products")
    if cur.fetchone()[0] == 0:
        products = [
            ('Apple iPhone', 'Electronics', 'Latest model smartphone.'),
            ('Denim Jeans', 'Clothing', 'Comfortable and stylish.'),
            ('Running Shoes', 'Footwear', 'Perfect for exercise.'),
            ('Wireless Headphones', 'Electronics', 'Noise-cancelling, bluetooth.'),
            ('Coffee Mug', 'Kitchen', 'Ceramic mug for coffee lovers.')
        ]
        cur.executemany("INSERT INTO products (name, category, description) VALUES (?, ?, ?)", products)
        conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/search')
def search():
    query = request.args.get('query', '').strip()
    conn = sqlite3.connect('shop.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if query:
        cur.execute("""
            SELECT * FROM products
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        """, (f'%{query}%', f'%{query}%', f'%{query}%'))
        results = cur.fetchall()
    else:
        results = []
    conn.close()
    return render_template_string(RESULTS_HTML, results=results, query=query)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
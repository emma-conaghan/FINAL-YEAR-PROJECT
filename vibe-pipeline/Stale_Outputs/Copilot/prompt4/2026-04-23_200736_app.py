from flask import Flask, render_template_string, request
import sqlite3

app = Flask(__name__)

HTML_HOME = """
<!doctype html>
<html>
<head>
    <title>Online Shop</title>
</head>
<body>
    <h1>Welcome to the Online Shop</h1>
    <form action="/search" method="get">
        <input type="text" name="query" placeholder="Search products..." required>
        <input type="submit" value="Search">
    </form>
</body>
</html>
"""

HTML_RESULTS = """
<!doctype html>
<html>
<head>
    <title>Search Results</title>
</head>
<body>
    <h1>Results for "{{ query }}"</h1>
    {% if results %}
        <ul>
        {% for product in results %}
            <li><strong>{{ product[1] }}</strong> - {{ product[2] }} - Category: {{ product[3] }}</li>
        {% endfor %}
        </ul>
    {% else %}
        <p>No products found.</p>
    {% endif %}
    <a href="/">Back to home</a>
</body>
</html>
"""

def init_db():
    conn = sqlite3.connect('shop.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            description TEXT,
            category TEXT
        )
    ''')
    c.execute('SELECT COUNT(*) FROM products')
    if c.fetchone()[0] == 0:
        products = [
            ('Laptop', 'A fast laptop', 'Electronics'),
            ('Headphones', 'Noise-cancelling headphones', 'Electronics'),
            ('Coffee Mug', 'Ceramic mug', 'Home'),
            ('Notebook', 'Spiral notebook', 'Stationery'),
            ('T-shirt', 'Cotton t-shirt', 'Apparel'),
            ('Novel', 'Mystery novel', 'Books'),
            ('Backpack', 'Travel backpack', 'Apparel'),
            ('Pen', 'Ballpoint pen', 'Stationery')
        ]
        c.executemany('INSERT INTO products (name, description, category) VALUES (?, ?, ?)', products)
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template_string(HTML_HOME)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    conn = sqlite3.connect('shop.db')
    c = conn.cursor()
    like_query = f"%{query}%"
    c.execute('''
        SELECT * FROM products WHERE
        name LIKE ? OR
        category LIKE ? OR
        description LIKE ?
    ''', (like_query, like_query, like_query))
    results = c.fetchall()
    conn.close()
    return render_template_string(HTML_RESULTS, query=query, results=results)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
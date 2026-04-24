from flask import Flask, request, render_template_string
import sqlite3

app = Flask(__name__)

HTML_INDEX = '''
<!doctype html>
<html>
<head><title>Online Shop Search</title></head>
<body>
<h1>Search Products</h1>
<form action="/search" method="get">
    <input type="text" name="query" placeholder="Search by name, category or keyword" required>
    <input type="submit" value="Search">
</form>
</body>
</html>
'''

HTML_RESULTS = '''
<!doctype html>
<html>
<head><title>Search Results</title></head>
<body>
<h1>Search Results</h1>
{% if products %}
<ul>
    {% for product in products %}
    <li><strong>{{ product[1] }}</strong> - Category: {{ product[2] }}<br>Description: {{ product[3] }}</li>
    {% endfor %}
</ul>
{% else %}
<p>No products found.</p>
{% endif %}
<a href="/">New Search</a>
</body>
</html>
'''

def get_db_connection():
    conn = sqlite3.connect('shop.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT
        )
    ''')
    c.execute('DELETE FROM products')  # Clear existing for demonstration
    products = [
        ('Apple iPhone', 'Electronics', 'Latest model smartphone'),
        ('Banana', 'Food', 'Fresh and organic'),
        ('Python Book', 'Books', 'Learn Python programming'),
        ('Headphones', 'Electronics', 'Noise cancelling headphones'),
        ('T-shirt', 'Clothing', 'Cotton T-shirt with logo'),
        ('Coffee Maker', 'Home Appliances', 'Brew coffee easily'),
    ]
    c.executemany('INSERT INTO products (name, category, description) VALUES (?, ?, ?)', products)
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template_string(HTML_INDEX)

@app.route('/search')
def search():
    query = request.args.get('query', '').strip()
    if not query:
        return render_template_string(HTML_RESULTS, products=[])
    conn = get_db_connection()
    c = conn.cursor()
    like_query = f'%{query.lower()}%'
    c.execute('''
        SELECT * FROM products
        WHERE lower(name) LIKE ?
        OR lower(category) LIKE ?
        OR lower(description) LIKE ?
    ''', (like_query, like_query, like_query))
    products = c.fetchall()
    conn.close()
    return render_template_string(HTML_RESULTS, products=products)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
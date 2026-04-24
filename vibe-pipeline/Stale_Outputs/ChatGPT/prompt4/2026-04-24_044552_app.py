from flask import Flask, request, render_template_string
import sqlite3

app = Flask(__name__)

# Initialize in-memory database and add sample data
def init_db():
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT
        )
    ''')
    products = [
        ("Red Shirt", "Clothing", "A bright red shirt, 100% cotton"),
        ("Blue Jeans", "Clothing", "Comfortable blue denim jeans"),
        ("Coffee Mug", "Kitchen", "Ceramic mug for coffee or tea"),
        ("Running Shoes", "Footwear", "Lightweight running shoes"),
        ("Wireless Mouse", "Electronics", "Battery powered mouse"),
        ("Bluetooth Speaker", "Electronics", "Portable speaker with great sound"),
        ("Notebook", "Stationery", "Ruled paper notebook"),
        ("Pen Set", "Stationery", "Set of 5 colorful pens")
    ]
    c.executemany("INSERT INTO products (name, category, description) VALUES (?, ?, ?)", products)
    conn.commit()
    return conn

conn = init_db()

SEARCH_HTML = '''
<!doctype html>
<title>Online Shop - Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
    <input type="text" name="query" placeholder="Search by name, category or keyword" size="40">
    <input type="submit" value="Search">
</form>
'''

RESULTS_HTML = '''
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
{% if results %}
    <ul>
    {% for product in results %}
        <li><strong>{{ product[1] }}</strong> (Category: {{ product[2] }})<br>{{ product[3] }}</li>
    {% endfor %}
    </ul>
{% else %}
    <p>No products found matching your search.</p>
{% endif %}
<a href="/">New Search</a>
'''

@app.route('/')
def home():
    return render_template_string(SEARCH_HTML)

@app.route('/results')
def results():
    query = request.args.get('query', '').strip()
    c = conn.cursor()
    if query:
        like_query = f'%{query}%'
        c.execute('''
            SELECT * FROM products 
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        ''', (like_query, like_query, like_query))
        items = c.fetchall()
    else:
        items = []
    return render_template_string(RESULTS_HTML, results=items)

if __name__ == '__main__':
    app.run(debug=True)
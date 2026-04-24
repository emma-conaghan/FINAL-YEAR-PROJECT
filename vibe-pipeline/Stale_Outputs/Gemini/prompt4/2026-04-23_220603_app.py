import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('shop.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS products')
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL
        )
    ''')
    
    sample_products = [
        ('Gaming Laptop', 'Electronics', 'High-end laptop with powerful GPU'),
        ('Wireless Mouse', 'Electronics', 'Ergonomic wireless optical mouse'),
        ('Cotton T-Shirt', 'Apparel', 'Comfortable 100% cotton plain shirt'),
        ('Running Shoes', 'Apparel', 'Lightweight sneakers for athletes'),
        ('Coffee Maker', 'Kitchen', 'Automatic drip coffee machine'),
        ('Stainless Steel Pot', 'Kitchen', 'Large cooking pot for family meals'),
        ('Desk Lamp', 'Home Office', 'Adjustable LED lamp with 3 modes'),
        ('Journal Notebook', 'Stationery', 'Leather-bound ruled notebook')
    ]
    
    cursor.executemany('INSERT INTO products (name, category, description) VALUES (?, ?, ?)', sample_products)
    conn.commit()
    conn.close()

init_db()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .product { border-bottom: 1px solid #ccc; padding: 10px 0; }
        .category { color: #666; font-style: italic; }
        .search-box { margin-bottom: 30px; background: #f4f4f4; padding: 20px; }
    </style>
</head>
<body>
    <h1>My Simple Shop</h1>
    
    <div class="search-box">
        <form action="/" method="GET">
            <input type="text" name="search" placeholder="Search by name, category, or keyword..." style="width: 300px;">
            <input type="submit" value="Search">
        </form>
    </div>

    {% if query %}
        <h3>Results for: "{{ query }}"</h3>
    {% else %}
        <h3>All Products</h3>
    {% endif %}

    <div class="results">
        {% if products %}
            {% for p in products %}
                <div class="product">
                    <strong>{{ p[1] }}</strong> - <span class="category">{{ p[2] }}</span>
                    <p>{{ p[3] }}</p>
                </div>
            {% endfor %}
        {% else %}
            <p>No products found matching your search.</p>
        {% endif %}
    </div>
    
    {% if query %}
        <p><a href="/">Show all products</a></p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    search_query = request.args.get('search', '').strip()
    conn = sqlite3.connect('shop.db')
    cursor = conn.cursor()
    
    if search_query:
        # Search name, category, or description for the keyword
        wildcard_query = f"%{search_query}%"
        cursor.execute('''
            SELECT * FROM products 
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        ''', (wildcard_query, wildcard_query, wildcard_query))
    else:
        cursor.execute('SELECT * FROM products')
    
    products = cursor.fetchall()
    conn.close()
    
    return render_template_string(HTML_PAGE, products=products, query=search_query)

if __name__ == '__main__':
    app.run(debug=True)
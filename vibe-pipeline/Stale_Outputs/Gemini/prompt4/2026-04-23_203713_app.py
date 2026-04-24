import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Initialize an in-memory database for the shop
def init_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            keyword TEXT
        )
    ''')
    
    # Add some sample data
    sample_products = [
        ('Gaming Laptop', 'Electronics', 'computer'),
        ('Wireless Mouse', 'Electronics', 'accessory'),
        ('Cotton T-Shirt', 'Clothing', 'apparel'),
        ('Coffee Mug', 'Kitchen', 'home'),
        ('Desk Lamp', 'Furniture', 'office'),
        ('Smartphone', 'Electronics', 'mobile'),
        ('Running Shoes', 'Clothing', 'footwear')
    ]
    cursor.executemany('INSERT INTO products (name, category, keyword) VALUES (?, ?, ?)', sample_products)
    conn.commit()
    return conn

db_connection = init_db()

# Basic HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .product-card { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        .search-box { margin-bottom: 30px; padding: 20px; background: #f4f4f4; border-radius: 5px; }
        input[type="text"] { padding: 8px; width: 250px; }
        input[type="submit"] { padding: 8px 15px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Mini Shop</h1>
    
    <div class="search-box">
        <h3>Search for Products</h3>
        <form action="/search" method="get">
            <input type="text" name="q" placeholder="Enter name, category, or keyword..." required>
            <input type="submit" value="Search">
        </form>
    </div>

    <h2>{{ title }}</h2>
    {% if products %}
        {% for product in products %}
        <div class="product-card">
            <strong>{{ product[1] }}</strong><br>
            Category: {{ product[2] }}<br>
            <small>Tags: {{ product[3] }}</small>
        </div>
        {% endfor %}
    {% else %}
        <p>No products found.</p>
    {% endif %}
    
    <p><a href="/">View All Products</a></p>
</body>
</html>
"""

@app.route('/')
def home():
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    return render_template_string(HTML_TEMPLATE, products=products, title="All Available Products")

@app.route('/search')
def search():
    query = request.args.get('q', '')
    cursor = db_connection.cursor()
    
    # Search logic: matches query against name, category, or keyword
    search_pattern = f"%{query}%"
    cursor.execute("""
        SELECT * FROM products 
        WHERE name LIKE ? OR category LIKE ? OR keyword LIKE ?
    """, (search_pattern, search_pattern, search_pattern))
    
    results = cursor.fetchall()
    return render_template_string(HTML_TEMPLATE, products=results, title=f"Search Results for '{query}'")

if __name__ == '__main__':
    # Start the Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)
import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Initialize in-memory database and seed data
db = sqlite3.connect(':memory:', check_same_thread=False)
cursor = db.cursor()
cursor.execute('CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, keyword TEXT, price REAL)')
sample_products = [
    (1, 'Pro Laptop', 'Electronics', 'computer portable work technology', 999.99),
    (2, 'Coffee Maker', 'Kitchen', 'brew morning caffeine appliance', 45.00),
    (3, 'Cotton T-Shirt', 'Apparel', 'clothing summer comfortable basic', 19.99),
    (4, 'Running Shoes', 'Footwear', 'sports exercise fitness gym', 65.00),
    (5, 'Bluetooth Headphones', 'Electronics', 'audio music wireless gadget', 89.00),
    (6, 'Ceramic Bowl', 'Kitchen', 'dining pottery home decor', 12.50)
]
cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?, ?)', sample_products)
db.commit()

# Single template for search box and results
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        .search-container { margin-bottom: 30px; background: #f4f4f4; padding: 20px; border-radius: 8px; }
        .product-card { border: 1px solid #eee; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
        .product-name { font-size: 1.2em; font-weight: bold; color: #333; }
        .product-cat { color: #666; font-size: 0.9em; }
        .product-price { color: #2c3e50; font-weight: bold; float: right; }
        input[type="text"] { width: 70%; padding: 10px; }
        input[type="submit"] { padding: 10px 20px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Mini Web Shop</h1>
    
    <div class="search-container">
        <form method="GET" action="/">
            <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query }}">
            <input type="submit" value="Search">
        </form>
    </div>

    {% if query %}
        <h2>Results for "{{ query }}"</h2>
        {% if results %}
            {% for p in results %}
                <div class="product-card">
                    <span class="product-price">${{ "%.2f"|format(p[4]) }}</span>
                    <div class="product-name">{{ p[1] }}</div>
                    <div class="product-cat">Category: {{ p[2] }}</div>
                    <div style="margin-top:5px;"><small>Tags: {{ p[3] }}</small></div>
                </div>
            {% endfor %}
        {% else %}
            <p>No products found matching your search.</p>
        {% endif %}
    {% else %}
        <p>Please enter a keyword to start searching our catalog.</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    query = request.args.get('q', '').strip()
    results = []
    
    if query:
        # Simple SQL search using LIKE for name, category, and keywords
        search_pattern = f"%{query}%"
        cursor.execute(
            "SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR keyword LIKE ?",
            (search_pattern, search_pattern, search_pattern)
        )
        results = cursor.fetchall()
        
    return render_template_string(HTML_TEMPLATE, query=query, results=results)

if __name__ == '__main__':
    # Running the application
    app.run(debug=True)
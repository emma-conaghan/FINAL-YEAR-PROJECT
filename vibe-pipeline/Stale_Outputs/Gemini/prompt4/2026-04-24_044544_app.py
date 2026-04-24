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
        ('Gaming Mouse', 'Electronics', 'High-performance optical wireless mouse'),
        ('Mechanical Keyboard', 'Electronics', 'RGB backlit mechanical keyboard'),
        ('Ceramic Coffee Mug', 'Kitchen', 'White ceramic mug with a 12oz capacity'),
        ('Cotton T-Shirt', 'Apparel', 'Comfortable 100% organic cotton shirt'),
        ('Noise Cancelling Headphones', 'Electronics', 'Wireless bluetooth over-ear headphones'),
        ('Yoga Mat', 'Fitness', 'Non-slip eco-friendly rubber yoga mat'),
        ('Steel Water Bottle', 'Kitchen', 'Insulated stainless steel water bottle')
    ]
    cursor.executemany('INSERT INTO products (name, category, description) VALUES (?, ?, ?)', sample_products)
    conn.commit()
    conn.close()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6; }
        .product-card { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px; }
        .category { color: #666; font-size: 0.9em; }
        input[type="text"] { padding: 10px; width: 300px; }
        input[type="submit"] { padding: 10px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Shop Inventory</h1>
    <form action="/search" method="get">
        <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query }}">
        <input type="submit" value="Search">
    </form>
    
    <hr>

    {% if results %}
        <h3>Found {{ results|length }} result(s):</h3>
        {% for product in results %}
            <div class="product-card">
                <strong>{{ product[1] }}</strong> <span class="category">({{ product[2] }})</span>
                <p>{{ product[3] }}</p>
            </div>
        {% endfor %}
    {% elif has_searched %}
        <p>No products found matching "{{ query }}".</p>
    {% else %}
        <p>Enter a keyword above to browse our products.</p>
    {% endif %}

    <p><a href="/">Clear Search</a></p>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, results=[], query="", has_searched=False)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    conn = sqlite3.connect('shop.db')
    cursor = conn.cursor()
    
    search_pattern = f"%{query}%"
    cursor.execute("""
        SELECT * FROM products 
        WHERE name LIKE ? 
        OR category LIKE ? 
        OR description LIKE ?
    """, (search_pattern, search_pattern, search_pattern))
    
    results = cursor.fetchall()
    conn.close()
    
    return render_template_string(
        HTML_TEMPLATE, 
        results=results, 
        query=query, 
        has_searched=True
    )

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
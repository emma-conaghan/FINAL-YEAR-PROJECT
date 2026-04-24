from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)
DATABASE = 'shop.db'


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    if os.path.exists(DATABASE):
        return
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            image_url TEXT
        )
    ''')

    sample_products = [
        ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 19.99, '🖱️'),
        ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 49.99, '⌨️'),
        ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI output', 29.99, '🔌'),
        ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 34.99, '💻'),
        ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 39.99, '📷'),
        ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 24.99, '💡'),
        ('Notebook Journal', 'Stationery', 'Hardcover lined notebook 200 pages', 12.99, '📓'),
        ('Ballpoint Pen Set', 'Stationery', 'Set of 10 colored ballpoint pens', 8.99, '🖊️'),
        ('Coffee Mug', 'Home Office', 'Ceramic coffee mug 350ml programmer design', 14.99, '☕'),
        ('Mouse Pad XL', 'Accessories', 'Extended mouse pad with stitched edges', 15.99, '🖥️'),
        ('Bluetooth Speaker', 'Electronics', 'Portable bluetooth speaker waterproof', 29.99, '🔊'),
        ('Phone Stand', 'Accessories', 'Adjustable phone holder for desk', 9.99, '📱'),
        ('Headphones', 'Electronics', 'Over-ear noise cancelling headphones', 59.99, '🎧'),
        ('Backpack', 'Accessories', 'Water-resistant laptop backpack 15 inch', 44.99, '🎒'),
        ('Sticky Notes', 'Stationery', 'Colorful sticky notes pack of 500', 6.99, '📝'),
        ('Desk Organizer', 'Home Office', 'Wooden desk organizer with multiple compartments', 22.99, '🗂️'),
        ('HDMI Cable', 'Electronics', 'High speed HDMI cable 2 meters', 11.99, '🔗'),
        ('Wrist Rest', 'Accessories', 'Memory foam keyboard wrist rest', 13.99, '🤲'),
        ('Plant Pot', 'Home Office', 'Small ceramic plant pot for desk succulent', 10.99, '🪴'),
        ('Book Light', 'Accessories', 'Rechargeable clip-on book reading light', 16.99, '📖'),
    ]

    cursor.executemany(
        'INSERT INTO products (name, category, description, price, image_url) VALUES (?, ?, ?, ?, ?)',
        sample_products
    )
    conn.commit()
    conn.close()


HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Online Shop</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .header {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 5px;
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .search-container {
            max-width: 700px;
            margin: 40px auto 20px auto;
            padding: 0 20px;
        }
        .search-box {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-box input[type="text"] {
            flex: 1;
            min-width: 200px;
            padding: 15px 20px;
            border: none;
            border-radius: 50px;
            font-size: 1.1em;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            outline: none;
        }
        .search-box input[type="text"]:focus {
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .search-box button {
            padding: 15px 30px;
            background: #ff6b6b;
            color: white;
            border: none;
            border-radius: 50px;
            font-size: 1.1em;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: background 0.3s;
        }
        .search-box button:hover {
            background: #ee5a5a;
        }
        .category-filter {
            max-width: 700px;
            margin: 15px auto;
            padding: 0 20px;
            text-align: center;
        }
        .category-filter a {
            display: inline-block;
            padding: 8px 18px;
            margin: 5px;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            border-radius: 20px;
            font-size: 0.95em;
            transition: background 0.3s;
        }
        .category-filter a:hover, .category-filter a.active {
            background: rgba(255,255,255,0.4);
        }
        .results-info {
            max-width: 900px;
            margin: 20px auto 10px auto;
            padding: 0 20px;
            color: white;
            font-size: 1.05em;
        }
        .products-grid {
            max-width: 900px;
            margin: 10px auto 40px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .product-icon {
            font-size: 3em;
            text-align: center;
            margin-bottom: 15px;
        }
        .product-name {
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }
        .product-category {
            display: inline-block;
            padding: 4px 12px;
            background: #e8f4f8;
            color: #2d8bba;
            border-radius: 12px;
            font-size: 0.85em;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 0.95em;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .product-price {
            font-size: 1.4em;
            font-weight: 700;
            color: #2d8bba;
        }
        .no-results {
            text-align: center;
            color: white;
            padding: 60px 20px;
            font-size: 1.3em;
        }
        .no-results .emoji {
            font-size: 3em;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛍️ Simple Online Shop</h1>
        <p>Search for products by name, category, or keyword</p>
    </div>

    <div class="search-container">
        <form class="search-box" action="/search" method="GET">
            <input type="text" name="q" placeholder="Search products..." value="{{ query or '' }}">
            <button type="submit">🔍 Search</button>
        </form>
    </div>

    <div class="category-filter">
        <a href="/" {% if not category %}class="active"{% endif %}>All</a>
        {% for cat in categories %}
        <a href="/search?category={{ cat }}" {% if category == cat %}class="active"{% endif %}>{{ cat }}</a>
        {% endfor %}
    </div>

    {% if products is defined %}
        <div class="results-info">
            {% if query %}
                Found {{ products|length }} result(s) for "{{ query }}"
            {% elif category %}
                Showing {{ products|length }} product(s) in "{{ category }}"
            {% else %}
                Showing all {{ products|length }} product(s)
            {% endif %}
        </div>

        {% if products|length > 0 %}
        <div class="products-grid">
            {% for product in products %}
            <div class="product-card">
                <div class="product-icon">{{ product.image_url }}</div>
                <div class="product-name">{{ product.name }}</div>
                <span class="product-category">{{ product.category }}</span>
                <div class="product-description">{{ product.description }}</div>
                <div class="product-price">${{ "%.2f"|format(product.price) }}</div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <div class="emoji">😕</div>
            <p>No products found. Try a different search term!</p>
        </div>
        {% endif %}
    {% endif %}
</body>
</html>
'''


@app.route('/')
def home():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]

    cursor.execute('SELECT * FROM products ORDER BY name')
    products = cursor.fetchall()
    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        products=products,
        categories=categories,
        query=None,
        category=None
    )


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    category = request.args.get('category', '').strip()

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]

    if query:
        search_term = f'%{query}%'
        cursor.execute(
            '''SELECT * FROM products
               WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
               ORDER BY name''',
            (search_term, search_term, search_term)
        )
    elif category:
        cursor.execute(
            'SELECT * FROM products WHERE category = ? ORDER BY name',
            (category,)
        )
    else:
        cursor.execute('SELECT * FROM products ORDER BY name')

    products = cursor.fetchall()
    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        products=products,
        categories=categories,
        query=query if query else None,
        category=category if category else None
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
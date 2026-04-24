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
            description TEXT NOT NULL,
            price REAL NOT NULL,
            image_url TEXT DEFAULT ''
        )
    ''')

    sample_products = [
        ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 24.99, '🖱️'),
        ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 79.99, '⌨️'),
        ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 34.99, '🔌'),
        ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for desk', 29.99, '💻'),
        ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with built-in microphone', 49.99, '📷'),
        ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 19.99, '💡'),
        ('Notebook Journal', 'Stationery', 'Premium leather-bound notebook 200 pages', 14.99, '📓'),
        ('Ballpoint Pen Set', 'Stationery', 'Set of 10 colored ballpoint pens', 9.99, '🖊️'),
        ('Coffee Mug', 'Home Office', 'Ceramic coffee mug with funny programmer quote', 12.99, '☕'),
        ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 15.99, '🖥️'),
        ('Bluetooth Speaker', 'Electronics', 'Portable bluetooth speaker waterproof', 39.99, '🔊'),
        ('Phone Stand', 'Accessories', 'Adjustable phone holder for desk', 11.99, '📱'),
        ('Headphone Stand', 'Accessories', 'Wooden headphone hanger for desk organization', 22.99, '🎧'),
        ('Screen Cleaner Kit', 'Accessories', 'Microfiber cloth and spray for screens', 8.99, '🧹'),
        ('Ergonomic Chair Cushion', 'Home Office', 'Memory foam seat cushion for office chair', 34.99, '🪑'),
        ('Wireless Charger', 'Electronics', 'Fast wireless charging pad for phones', 19.99, '🔋'),
        ('Sticky Notes Pack', 'Stationery', 'Colorful sticky notes 500 sheets assorted', 6.99, '📝'),
        ('Cable Organizer', 'Accessories', 'Silicone cable management clips set of 5', 7.99, '🔗'),
        ('Monitor Light Bar', 'Home Office', 'LED monitor light bar with auto dimming', 44.99, '🌟'),
        ('Typing Wrist Rest', 'Accessories', 'Gel wrist rest pad for keyboard comfort', 13.99, '🤚'),
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
    <title>Mini Online Shop</title>
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
            color: #333;
        }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 20px 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 28px;
            color: #667eea;
        }
        .header h1 span {
            color: #764ba2;
        }
        .nav-links a {
            text-decoration: none;
            color: #667eea;
            margin-left: 20px;
            font-weight: 600;
        }
        .nav-links a:hover {
            color: #764ba2;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .search-section {
            text-align: center;
            margin-bottom: 40px;
        }
        .search-section h2 {
            color: white;
            font-size: 36px;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .search-section p {
            color: rgba(255,255,255,0.85);
            font-size: 18px;
            margin-bottom: 30px;
        }
        .search-form {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            padding: 14px 24px;
            font-size: 16px;
            border: none;
            border-radius: 50px;
            width: 400px;
            max-width: 90vw;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            outline: none;
        }
        .search-form input[type="text"]:focus {
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }
        .search-form select {
            padding: 14px 24px;
            font-size: 16px;
            border: none;
            border-radius: 50px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            outline: none;
            background: white;
            cursor: pointer;
        }
        .search-form button {
            padding: 14px 32px;
            font-size: 16px;
            border: none;
            border-radius: 50px;
            background: #ff6b6b;
            color: white;
            cursor: pointer;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(255,107,107,0.4);
            transition: all 0.3s;
        }
        .search-form button:hover {
            background: #ee5a5a;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255,107,107,0.5);
        }
        .results-info {
            color: white;
            text-align: center;
            margin-bottom: 20px;
            font-size: 18px;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 24px;
        }
        .product-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .product-emoji {
            font-size: 48px;
            margin-bottom: 12px;
        }
        .product-name {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 6px;
            color: #333;
        }
        .product-category {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 14px;
            line-height: 1.5;
            margin-bottom: 12px;
        }
        .product-price {
            font-size: 24px;
            font-weight: 700;
            color: #ff6b6b;
        }
        .no-results {
            text-align: center;
            color: white;
            padding: 60px 20px;
        }
        .no-results h3 {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .no-results p {
            font-size: 16px;
            opacity: 0.8;
        }
        .categories-section {
            margin-top: 40px;
            text-align: center;
        }
        .categories-section h3 {
            color: white;
            margin-bottom: 15px;
            font-size: 20px;
        }
        .category-tags {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .category-tag {
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 8px 20px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s;
            border: 2px solid rgba(255,255,255,0.3);
        }
        .category-tag:hover {
            background: rgba(255,255,255,0.3);
            border-color: rgba(255,255,255,0.5);
        }
        .footer {
            text-align: center;
            color: rgba(255,255,255,0.6);
            padding: 40px 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛒 Mini <span>Shop</span></h1>
        <div class="nav-links">
            <a href="/">Home</a>
            <a href="/search">Browse All</a>
        </div>
    </div>

    <div class="container">
        <div class="search-section">
            <h2>Find What You Need</h2>
            <p>Search our collection of products by name, category, or keyword</p>
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search for products..." value="{{ query or '' }}">
                <select name="category">
                    <option value="">All Categories</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                    {% endfor %}
                </select>
                <button type="submit">🔍 Search</button>
            </form>
        </div>

        {% if searched %}
            <div class="results-info">
                {% if products %}
                    Found <strong>{{ products|length }}</strong> product(s)
                    {% if query %} matching "<strong>{{ query }}</strong>"{% endif %}
                    {% if selected_category %} in <strong>{{ selected_category }}</strong>{% endif %}
                {% endif %}
            </div>

            {% if products %}
                <div class="products-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-emoji">{{ product.image_url }}</div>
                        <div class="product-name">{{ product.name }}</div>
                        <div class="product-category">{{ product.category }}</div>
                        <div class="product-description">{{ product.description }}</div>
                        <div class="product-price">${{ "%.2f"|format(product.price) }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <h3>😕 No products found</h3>
                    <p>Try a different search term or browse all categories</p>
                </div>
            {% endif %}
        {% else %}
            <div class="categories-section">
                <h3>Browse by Category</h3>
                <div class="category-tags">
                    {% for cat in categories %}
                    <a class="category-tag" href="/search?category={{ cat }}">{{ cat }}</a>
                    {% endfor %}
                </div>
            </div>

            {% if products %}
                <div class="results-info" style="margin-top: 40px;">
                    <strong>Featured Products</strong>
                </div>
                <div class="products-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-emoji">{{ product.image_url }}</div>
                        <div class="product-name">{{ product.name }}</div>
                        <div class="product-category">{{ product.category }}</div>
                        <div class="product-description">{{ product.description }}</div>
                        <div class="product-price">${{ "%.2f"|format(product.price) }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endif %}
    </div>

    <div class="footer">
        <p>&copy; 2024 Mini Shop - A simple online shop built with Flask</p>
    </div>
</body>
</html>
'''


def get_categories():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]
    conn.close()
    return categories


def get_featured_products(limit=6):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products ORDER BY id LIMIT ?', (limit,))
    products = cursor.fetchall()
    conn.close()
    return products


def search_products(query='', category=''):
    conn = get_db()
    cursor = conn.cursor()

    conditions = []
    params = []

    if query:
        conditions.append('(name LIKE ? OR description LIKE ? OR category LIKE ?)')
        search_term = f'%{query}%'
        params.extend([search_term, search_term
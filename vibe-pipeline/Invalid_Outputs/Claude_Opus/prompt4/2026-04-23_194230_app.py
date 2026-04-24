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
        ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 19.99, '🖱️'),
        ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 49.99, '⌨️'),
        ('USB-C Hub', 'Electronics', 'Multi-port USB-C adapter with HDMI output', 29.99, '🔌'),
        ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better posture', 34.99, '💻'),
        ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with built-in microphone', 39.99, '📷'),
        ('Noise Cancelling Headphones', 'Audio', 'Over-ear wireless headphones with active noise cancellation', 89.99, '🎧'),
        ('Bluetooth Speaker', 'Audio', 'Portable waterproof Bluetooth speaker', 24.99, '🔊'),
        ('Phone Case', 'Accessories', 'Shockproof silicone phone case, multiple colors', 9.99, '📱'),
        ('Screen Protector', 'Accessories', 'Tempered glass screen protector for smartphones', 7.99, '🛡️'),
        ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness and color temperature', 22.99, '💡'),
        ('Mouse Pad XL', 'Accessories', 'Extra large gaming mouse pad with stitched edges', 12.99, '🎮'),
        ('Cable Organizer', 'Home Office', 'Silicone cable management clips for desk organization', 5.99, '📎'),
        ('Portable Charger', 'Electronics', 'Power bank 10000mAh portable charger with fast charging', 19.99, '🔋'),
        ('Earbuds', 'Audio', 'True wireless earbuds with charging case', 29.99, '🎵'),
        ('Monitor Light Bar', 'Home Office', 'LED monitor light bar to reduce eye strain', 44.99, '🖥️'),
        ('Tablet Stylus', 'Accessories', 'Universal capacitive stylus pen for tablets and phones', 14.99, '✏️'),
        ('HDMI Cable', 'Electronics', 'High speed HDMI 2.1 cable 6 feet braided', 11.99, '🔗'),
        ('Desk Organizer', 'Home Office', 'Wooden desk organizer with multiple compartments', 18.99, '🗂️'),
        ('Wrist Rest', 'Accessories', 'Memory foam keyboard wrist rest pad', 13.99, '🤲'),
        ('Smart Plug', 'Electronics', 'WiFi smart plug compatible with voice assistants', 15.99, '🔌'),
    ]

    cursor.executemany(
        'INSERT INTO products (name, category, description, price, image_url) VALUES (?, ?, ?, ?, ?)',
        sample_products
    )
    conn.commit()
    conn.close()


def search_products(query):
    conn = get_db()
    cursor = conn.cursor()
    if not query or query.strip() == '':
        cursor.execute('SELECT * FROM products ORDER BY name')
    else:
        search_term = f'%{query}%'
        cursor.execute('''
            SELECT * FROM products 
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
            ORDER BY name
        ''', (search_term, search_term, search_term))
    results = cursor.fetchall()
    conn.close()
    return results


def get_categories():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]
    conn.close()
    return categories


BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini Shop - Online Store</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            min-height: 100vh;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 15px;
        }
        .logo {
            font-size: 28px;
            font-weight: bold;
            text-decoration: none;
            color: white;
        }
        .logo span {
            font-size: 32px;
        }
        .search-form {
            display: flex;
            gap: 8px;
            flex-grow: 1;
            max-width: 500px;
        }
        .search-form input[type="text"] {
            flex-grow: 1;
            padding: 12px 18px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .search-form input[type="text"]:focus {
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .search-form button {
            padding: 12px 24px;
            border: 2px solid white;
            background: transparent;
            color: white;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .search-form button:hover {
            background: white;
            color: #667eea;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .categories {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 25px;
        }
        .category-link {
            display: inline-block;
            padding: 8px 18px;
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
            border-radius: 20px;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .category-link:hover, .category-link.active {
            background: #667eea;
            color: white;
        }
        .results-info {
            margin-bottom: 20px;
            font-size: 16px;
            color: #666;
        }
        .results-info strong {
            color: #333;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }
        .product-emoji {
            font-size: 48px;
            text-align: center;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9ff;
            border-radius: 10px;
        }
        .product-category {
            display: inline-block;
            padding: 4px 12px;
            background: #e8ecff;
            color: #667eea;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .product-name {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #222;
        }
        .product-description {
            font-size: 14px;
            color: #777;
            margin-bottom: 15px;
            flex-grow: 1;
            line-height: 1.5;
        }
        .product-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .product-price {
            font-size: 22px;
            font-weight: 800;
            color: #667eea;
        }
        .add-to-cart-btn {
            padding: 8px 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: opacity 0.3s;
        }
        .add-to-cart-btn:hover {
            opacity: 0.85;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .no-results h2 {
            font-size: 24px;
            margin-bottom: 10px;
            color: #666;
        }
        .no-results p {
            font-size: 16px;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #999;
            margin-top: 40px;
            font-size: 14px;
        }
        @media (max-width: 600px) {
            .header-content {
                flex-direction: column;
                text-align: center;
            }
            .search-form {
                max-width: 100%;
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <a href="/" class="logo"><span>🛍️</span> Mini Shop</a>
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search products..." value="{{ query if query else '' }}">
                <button type="submit">Search</button>
            </form>
        </div>
    </header>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
    <footer>
        &copy; 2024 Mini Shop. A simple online store built with Flask &amp; SQLite.
    </footer>
</body>
</html>
'''

HOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <h2 style="margin-bottom: 15px; color: #444;">Browse by Category</h2>
    <div class="categories">
        <a href="/search?q=" class="category-link active">All Products</a>
        {% for cat in categories %}
            <a href="/search?q={{ cat }}" class="category-link">{{ cat }}</a>
        {% endfor %}
    </div>
    <div class="results-info">
        Showing <strong>{{ products|length }}</strong> product{{ 's' if products|length != 1 else '' }}
    </div>
    <div class="products-grid">
        {% for product in products %}
        <div class="product-card">
            <div class="product-emoji">{{ product['image_url'] }}</div>
            <span class="product-category">{{ product['category'] }}</span>
            <div class="product-name">{{ product['name'] }}</div>
            <div class="product-description">{{ product['description'] }}</div>
            <div class="product-footer">
                <span class="product-price">${{ "%.2f"|format(product['price']) }}</span>
                <button class="add-to-cart-btn" onclick="alert('{{ product['name'] }} added to cart!')">Add to Cart</button>
            </div>
        </div>
        {% endfor %}
    </div>
    {% if products|length == 0 %}
    <div class="no-results">
        <h2>No products found</h2>
        <p>Try a different search term or browse our categories above.</p>
    </div>
    {% endif %}
{% endblock %}
'''

SEARCH_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <h2 style="margin-bottom: 15px; color: #444;">Browse by Category</h2>
    <div class="categories">
        <a href="/" class="category-link">All Products</a>
        {% for cat in categories %}
            <a href="/search?q={{ cat }}" class="category-link {{ 'active' if query == cat else '' }}">{{ cat }}</a>
        {% endfor %}
    </div>
    <div class="results-info">
        {% if query %}
            Search results for "<strong>{{ query }}</strong>" — <strong>{{ products|length }}</strong> product{{ 's' if products|length != 1 else '' }} found
        {% else %}
            Showing <strong>{{ products|length }}</strong> product{{ 's' if products|length != 1 else '' }}
        {% endif %}
    </div>
    <div class="products-grid">
        {% for product in products %}
        <div class="product-card">
            <div class="product-emoji">{{ product['image_url'] }}</div>
            <span class="product-category">{{ product['category'] }}</span>
            <div class="product-name">{{ product['name'] }}</div>
            <div class="product-description">{{ product['description'] }}</div>
            <div class="product-footer">
                <span class="product-price">${{ "%.2f"|format(product['price']) }}</span>
                <button class="add-to
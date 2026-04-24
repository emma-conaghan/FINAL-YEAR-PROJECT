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
        ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI output', 34.99, '🔌'),
        ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for ergonomic setup', 45.99, '💻'),
        ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with built-in microphone', 49.99, '📷'),
        ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness levels', 29.99, '💡'),
        ('Notebook Journal', 'Stationery', 'Premium hardcover notebook with 200 lined pages', 12.99, '📓'),
        ('Ballpoint Pen Set', 'Stationery', 'Set of 10 smooth writing ballpoint pens', 8.99, '🖊️'),
        ('Coffee Mug', 'Home Office', 'Large ceramic coffee mug with funny coding quote', 14.99, '☕'),
        ('Mouse Pad', 'Accessories', 'Extra large mouse pad with non-slip rubber base', 15.99, '🖥️'),
        ('Headphones', 'Electronics', 'Over-ear noise cancelling headphones with Bluetooth', 99.99, '🎧'),
        ('Phone Charger', 'Electronics', 'Fast charging USB-C phone charger cable', 12.99, '🔋'),
        ('Backpack', 'Accessories', 'Water-resistant laptop backpack with USB charging port', 55.99, '🎒'),
        ('Sticky Notes', 'Stationery', 'Colorful sticky notes pack of 500 sheets', 6.99, '📝'),
        ('Monitor Stand', 'Home Office', 'Wooden monitor stand with storage drawer', 39.99, '🖥️'),
        ('Water Bottle', 'Accessories', 'Insulated stainless steel water bottle 750ml', 19.99, '🍶'),
        ('Desk Organizer', 'Home Office', 'Bamboo desk organizer with multiple compartments', 22.99, '📦'),
        ('Wireless Earbuds', 'Electronics', 'True wireless earbuds with charging case', 59.99, '🎵'),
        ('Blue Light Glasses', 'Accessories', 'Computer glasses that block blue light for eye protection', 18.99, '👓'),
        ('Whiteboard', 'Home Office', 'Magnetic dry erase whiteboard 24x36 inches', 34.99, '📋'),
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
    search_term = f'%{query}%'
    cursor.execute('''
        SELECT * FROM products 
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        ORDER BY name
    ''', (search_term, search_term, search_term))
    results = cursor.fetchall()
    conn.close()
    return results


def get_all_products():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products ORDER BY name')
    results = cursor.fetchall()
    conn.close()
    return results


def get_categories():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    results = [row['category'] for row in cursor.fetchall()]
    conn.close()
    return results


BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - Simple Online Shop</title>
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
            line-height: 1.6;
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
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        .logo {
            font-size: 28px;
            font-weight: bold;
            text-decoration: none;
            color: white;
        }
        .logo:hover {
            opacity: 0.9;
        }
        nav a {
            color: white;
            text-decoration: none;
            margin-left: 20px;
            font-size: 16px;
            opacity: 0.9;
        }
        nav a:hover {
            opacity: 1;
            text-decoration: underline;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-section {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            margin: 30px 0;
            text-align: center;
        }
        .search-section h2 {
            margin-bottom: 20px;
            color: #444;
            font-size: 24px;
        }
        .search-form {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-input {
            padding: 14px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            width: 400px;
            max-width: 100%;
            outline: none;
            transition: border-color 0.3s;
        }
        .search-input:focus {
            border-color: #667eea;
        }
        .search-button {
            padding: 14px 30px;
            font-size: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .search-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .categories {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .category-tag {
            display: inline-block;
            padding: 8px 16px;
            background: #f0f0f0;
            border-radius: 20px;
            text-decoration: none;
            color: #555;
            font-size: 14px;
            transition: background 0.3s, color 0.3s;
        }
        .category-tag:hover {
            background: #667eea;
            color: white;
        }
        .results-info {
            margin: 20px 0;
            font-size: 16px;
            color: #666;
        }
        .results-info strong {
            color: #333;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }
        .product-emoji {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .product-name {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        .product-category {
            display: inline-block;
            padding: 4px 12px;
            background: #e8f0fe;
            color: #667eea;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .product-price {
            font-size: 22px;
            font-weight: bold;
            color: #2d8f2d;
        }
        .product-actions {
            margin-top: 15px;
        }
        .add-to-cart-btn {
            padding: 10px 20px;
            background: #2d8f2d;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: background 0.3s;
        }
        .add-to-cart-btn:hover {
            background: #247a24;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #888;
        }
        .no-results h3 {
            font-size: 24px;
            margin-bottom: 10px;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #888;
            margin-top: 40px;
            font-size: 14px;
        }
        .hero {
            text-align: center;
            padding: 40px 20px;
        }
        .hero h1 {
            font-size: 36px;
            color: #333;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 18px;
            color: #666;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <a href="/" class="logo">🛒 Simple Shop</a>
            <nav>
                <a href="/">Home</a>
                <a href="/products">All Products</a>
            </nav>
        </div>
    </header>
    <div class="container">
        {{ content }}
    </div>
    <footer>
        <p>&copy; 2024 Simple Online Shop. Built with Flask &amp; Python.</p>
    </footer>
</body>
</html>
'''

HOME_CONTENT = '''
{% set content %}
<div class="hero">
    <h1>Welcome to Simple Shop</h1>
    <p>Find the best products for your home office and more!</p>
</div>

<div class="search-section">
    <h2>🔍 Search for Products</h2>
    <form action="/search" method="GET" class="search-form">
        <input type="text" name="q" class="search-input" placeholder="Search by name, category, or keyword..." 
               value="{{ query or '' }}" autofocus>
        <button type="submit" class="search-button">Search</button>
    </form>
    <div class="categories">
        <span style="color: #888; padding: 8px;">Quick categories:</span>
        {% for cat in categories %}
        <a href="/search?q={{ cat }}" class="category-tag">{{ cat }}</a>
        {% endfor %}
    </div>
</div>

<h2 style="margin-top: 30px; color: #444;">⭐ Featured Products</h2>
<div class="products-grid">
    {% for product in featured %}
    <div class="product-card">
        <div class="product-emoji">{{ product['image_url'] }}</div>
        <div class="product-name">{{ product['name'] }}</div>
        <span class="product-category">{{ product['category'] }}</span>
        <div class="product-description">{{ product['description'] }}</div>
        <div class="product-price">${{ "%.2f" | format(product['price']) }}</div>
        <div class="product-actions">
            <button class="add-to-cart-btn" onclick="alert('Added {{ product['name'] }} to cart!')">Add to Cart</button>
        </div>
    </div>
    {% endfor %}
</div>
{% endset %}
''' + BASE_TEMPLATE

SEARCH_CONTENT = '''
{% set content %}
<div class="search-section">
    <h2>🔍 Search Products</h2>
    <form action="/search" method="GET" class="search-form">
        <input type="text" name="q" class="search-input" placeholder="Search by name, category, or keyword..." 
               value="{{ query or '' }}" autofocus>
        <button type="submit" class="search-button">Search</button>
    </form>
    <div class="categories">
        <span style="color: #888; padding: 8px;">Quick categories:</span>
        {% for cat in categories %}
        <a href="/search?q={{ cat }}" class="category-tag">{{ cat }}</a>
        {% endfor %}
    </div>
</div>

{% if query %}
<div class="results-info">
    {% if products | length > 0 %}
    Found <strong>{{ products | length }}</strong> result(s) for "<strong>{{ query
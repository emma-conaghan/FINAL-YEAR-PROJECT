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
        ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 45.99, '💻'),
        ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with microphone', 49.99, '📷'),
        ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 29.99, '💡'),
        ('Notebook Journal', 'Stationery', 'Leather-bound notebook with 200 pages', 15.99, '📓'),
        ('Ballpoint Pen Set', 'Stationery', 'Premium ballpoint pen set of 5 colors', 9.99, '🖊️'),
        ('Coffee Mug', 'Home Office', 'Ceramic coffee mug with funny coding quotes', 12.99, '☕'),
        ('Monitor Stand', 'Accessories', 'Wooden monitor stand with storage drawer', 39.99, '🖥️'),
        ('Bluetooth Speaker', 'Electronics', 'Portable bluetooth speaker with bass boost', 55.99, '🔊'),
        ('Mouse Pad XL', 'Accessories', 'Extended mouse pad for gaming and work', 19.99, '🎮'),
        ('Phone Stand', 'Accessories', 'Adjustable phone stand for desk', 14.99, '📱'),
        ('Headphones', 'Electronics', 'Over-ear noise cancelling headphones', 89.99, '🎧'),
        ('Desk Organizer', 'Home Office', 'Bamboo desk organizer with multiple compartments', 27.99, '📦'),
        ('Sticky Notes', 'Stationery', 'Colorful sticky notes pack of 12 pads', 7.99, '📝'),
        ('Water Bottle', 'Accessories', 'Insulated stainless steel water bottle 750ml', 22.99, '🧴'),
        ('Cable Clips', 'Home Office', 'Adhesive cable clips organizer set of 10', 8.99, '📎'),
        ('Wireless Charger', 'Electronics', 'Fast wireless charging pad for smartphones', 29.99, '🔋'),
        ('Backpack', 'Accessories', 'Laptop backpack with USB charging port', 59.99, '🎒'),
    ]
    cursor.executemany(
        'INSERT INTO products (name, category, description, price, image_url) VALUES (?, ?, ?, ?, ?)',
        sample_products
    )
    conn.commit()
    conn.close()


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
            background-color: #f0f2f5;
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
            padding: 30px 20px;
        }
        .search-section {
            background: white;
            border-radius: 12px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
        }
        .search-section h1 {
            font-size: 32px;
            margin-bottom: 10px;
            color: #333;
        }
        .search-section p {
            color: #666;
            margin-bottom: 25px;
            font-size: 16px;
        }
        .search-form {
            display: flex;
            gap: 10px;
            max-width: 700px;
            margin: 0 auto;
            flex-wrap: wrap;
            justify-content: center;
        }
        .search-form input[type="text"] {
            flex: 1;
            min-width: 250px;
            padding: 14px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        .search-form input[type="text"]:focus {
            border-color: #667eea;
        }
        .search-form select {
            padding: 14px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            outline: none;
            background: white;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        .search-form select:focus {
            border-color: #667eea;
        }
        .search-form button {
            padding: 14px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: opacity 0.3s, transform 0.2s;
            font-weight: 600;
        }
        .search-form button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        .results-info {
            margin-bottom: 20px;
            color: #666;
            font-size: 16px;
        }
        .results-info strong {
            color: #333;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }
        .product-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .product-name {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        .product-category {
            display: inline-block;
            background: #f0f2f5;
            color: #667eea;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 14px;
            line-height: 1.5;
            margin-bottom: 15px;
        }
        .product-price {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .no-results h2 {
            font-size: 24px;
            color: #333;
            margin-bottom: 10px;
        }
        .no-results p {
            color: #666;
            font-size: 16px;
        }
        .categories {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 20px;
        }
        .category-tag {
            display: inline-block;
            padding: 8px 18px;
            background: #f0f2f5;
            color: #555;
            border-radius: 20px;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.3s, color 0.3s;
        }
        .category-tag:hover {
            background: #667eea;
            color: white;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #999;
            font-size: 14px;
        }
        .product-detail-section {
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        .detail-icon {
            font-size: 80px;
            margin-bottom: 20px;
        }
        .detail-name {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .detail-price {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 20px;
        }
        .detail-description {
            font-size: 18px;
            color: #555;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        .add-to-cart-btn {
            padding: 14px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            font-weight: 600;
            transition: opacity 0.3s;
        }
        .add-to-cart-btn:hover {
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <a href="/" class="logo">🛍️ SimpleShop</a>
            <nav>
                <a href="/">Home</a>
                <a href="/search">Search</a>
                <a href="/categories">Categories</a>
            </nav>
        </div>
    </header>
    {{ content }}
    <footer>
        &copy; 2024 SimpleShop - A beginner-friendly online shop demo
    </footer>
</body>
</html>
'''

HOME_CONTENT = '''
<div class="container">
    <div class="search-section">
        <h1>Welcome to SimpleShop 🛍️</h1>
        <p>Find the best products at great prices. Search by name, category, or keyword!</p>
        <form class="search-form" action="/search" method="GET">
            <input type="text" name="q" placeholder="Search for products..." value="">
            <button type="submit">🔍 Search</button>
        </form>
        <div class="categories">
            <span style="color: #999; padding: 8px 0;">Popular categories:</span>
            {% for cat in categories %}
            <a href="/search?category={{ cat }}" class="category-tag">{{ cat }}</a>
            {% endfor %}
        </div>
    </div>
    <div class="results-info">
        <strong>Featured Products</strong> — {{ products|length }} items available
    </div>
    <div class="product-grid">
        {% for product in products %}
        <a href="/product/{{ product.id }}" style="text-decoration: none; color: inherit;">
            <div class="product-card">
                <div class="product-icon">{{ product.image_url }}</div>
                <div class="product-name">{{ product.name }}</div>
                <span class="product-category">{{ product.category }}</span>
                <div class="product-description">{{ product.description }}</div>
                <div class="product-price">${{ "%.2f"|format(product.price) }}</div>
            </div>
        </a>
        {% endfor %}
    </div>
</div>
'''

SEARCH_CONTENT = '''
<div class="container">
    <div class="search-section">
        <h1>Search Products</h1>
        <p>Find exactly what you're looking for</p>
        <form class="search-form" action="/search" method="GET">
            <input type="text" name="q" placeholder="Search by name or keyword..." value="{{ query }}">
            <select name="category">
                <option value="">All Categories</option>
                {% for cat in categories %}
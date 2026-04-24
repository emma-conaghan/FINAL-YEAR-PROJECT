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
        ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better posture', 45.99, '💻'),
        ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with built-in microphone', 49.99, '📷'),
        ('Noise Cancelling Headphones', 'Audio', 'Over-ear Bluetooth headphones with active noise cancelling', 149.99, '🎧'),
        ('Portable Speaker', 'Audio', 'Waterproof portable Bluetooth speaker', 39.99, '🔊'),
        ('Phone Case', 'Accessories', 'Shockproof silicone phone case - universal fit', 12.99, '📱'),
        ('Screen Protector', 'Accessories', 'Tempered glass screen protector pack of 3', 9.99, '🛡️'),
        ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness and color temperature', 29.99, '💡'),
        ('Monitor Arm', 'Home Office', 'Single monitor arm mount with full motion swivel', 54.99, '🖥️'),
        ('Cable Organizer', 'Home Office', 'Cable management clips and ties set', 7.99, '🔗'),
        ('Mousepad XL', 'Accessories', 'Extra large gaming mousepad with stitched edges', 14.99, '🎮'),
        ('Flash Drive 64GB', 'Storage', 'High speed USB 3.0 flash drive 64GB', 11.99, '💾'),
        ('External SSD 500GB', 'Storage', 'Portable external SSD with fast read write speeds', 59.99, '💿'),
        ('Wireless Charger', 'Electronics', 'Fast wireless charging pad compatible with all Qi devices', 19.99, '🔋'),
        ('Smart Plug', 'Smart Home', 'WiFi smart plug with voice control support', 14.99, '🔌'),
        ('Fitness Tracker', 'Wearables', 'Waterproof fitness band with heart rate monitor', 39.99, '⌚'),
        ('Earbuds', 'Audio', 'True wireless earbuds with charging case', 29.99, '🎵'),
        ('Backpack', 'Bags', 'Water resistant laptop backpack with USB charging port', 44.99, '🎒'),
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
            justify-content: space-between;
            align-items: center;
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
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-left: 20px;
            font-size: 16px;
            opacity: 0.9;
        }
        .nav-links a:hover {
            opacity: 1;
            text-decoration: underline;
        }
        .hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px 60px;
            text-align: center;
        }
        .hero h1 {
            font-size: 36px;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        .search-container {
            max-width: 600px;
            margin: 0 auto;
            position: relative;
        }
        .search-form {
            display: flex;
            gap: 10px;
        }
        .search-input {
            flex: 1;
            padding: 15px 20px;
            font-size: 16px;
            border: none;
            border-radius: 50px;
            outline: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .search-button {
            padding: 15px 30px;
            font-size: 16px;
            background-color: #ff6b6b;
            color: white;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .search-button:hover {
            background-color: #ee5a24;
        }
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .categories {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 30px;
            justify-content: center;
        }
        .category-tag {
            display: inline-block;
            padding: 8px 18px;
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
            border-radius: 25px;
            text-decoration: none;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .category-tag:hover, .category-tag.active {
            background: #667eea;
            color: white;
        }
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .results-header h2 {
            font-size: 24px;
            color: #333;
        }
        .results-count {
            color: #666;
            font-size: 16px;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
        }
        .product-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .product-emoji {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            height: 150px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 64px;
        }
        .product-info {
            padding: 20px;
        }
        .product-category {
            display: inline-block;
            padding: 4px 12px;
            background: #e8f0fe;
            color: #667eea;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .product-name {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #333;
        }
        .product-description {
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .product-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .product-price {
            font-size: 22px;
            font-weight: bold;
            color: #667eea;
        }
        .add-to-cart {
            padding: 8px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background 0.3s;
        }
        .add-to-cart:hover {
            background: #764ba2;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .no-results .emoji {
            font-size: 64px;
            margin-bottom: 20px;
        }
        .no-results h3 {
            font-size: 24px;
            margin-bottom: 10px;
            color: #333;
        }
        footer {
            text-align: center;
            padding: 30px 20px;
            color: #999;
            font-size: 14px;
            margin-top: 40px;
        }
        @media (max-width: 600px) {
            .hero h1 {
                font-size: 24px;
            }
            .search-form {
                flex-direction: column;
            }
            .search-input, .search-button {
                border-radius: 12px;
            }
            .header-content {
                flex-direction: column;
                gap: 10px;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <a href="/" class="logo"><span>🛒</span> Mini Shop</a>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/search">All Products</a>
            </div>
        </div>
    </header>

    <div class="hero">
        <h1>Welcome to Mini Online Shop</h1>
        <p>Find the best products at amazing prices</p>
        <div class="search-container">
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" class="search-input" 
                       placeholder="Search products by name, category, or keyword..."
                       value="{{ query if query else '' }}">
                <button type="submit" class="search-button">🔍 Search</button>
            </form>
        </div>
    </div>

    <div class="main-content">
        <div class="categories">
            <a href="/search" class="category-tag {% if not query %}active{% endif %}">All</a>
            {% for cat in categories %}
            <a href="/search?q={{ cat }}" class="category-tag {% if query == cat %}active{% endif %}">{{ cat }}</a>
            {% endfor %}
        </div>

        {% if show_results %}
        <div class="results-header">
            <h2>
                {% if query %}
                    Results for "{{ query }}"
                {% else %}
                    All Products
                {% endif %}
            </h2>
            <span class="results-count">{{ products|length }} product(s) found</span>
        </div>

        {% if products %}
        <div class="products-grid">
            {% for product in products %}
            <div class="product-card">
                <div class="product-emoji">{{ product['image_url'] }}</div>
                <div class="product-info">
                    <span class="product-category">{{ product['category'] }}</span>
                    <div class="product-name">{{ product['name'] }}</div>
                    <div class="product-description">{{ product['description'] }}</div>
                    <div class="product-footer">
                        <span class="product-price">${{ "%.2f"|format(product['price']) }}</span>
                        <button class="add-to-cart" onclick="alert('{{ product['name'] }} added to cart!')">Add to Cart</button>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <div class="emoji">😕</div>
            <h3>No products found</h3>
            <p>Try a different search term or browse our categories above.</p>
        </div>
        {% endif %}
        {% else %}
        <div class
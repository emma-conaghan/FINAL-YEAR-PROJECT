from flask import Flask, render_template_string, request
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
        ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 29.99, '🖱️'),
        ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 79.99, '⌨️'),
        ('USB-C Hub', 'Electronics', 'Multi-port USB-C adapter with HDMI output', 49.99, '🔌'),
        ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better posture', 39.99, '💻'),
        ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with built-in microphone', 59.99, '📷'),
        ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness and color temperature', 34.99, '💡'),
        ('Notebook Journal', 'Stationery', 'Premium lined notebook with hardcover binding', 12.99, '📓'),
        ('Ballpoint Pen Set', 'Stationery', 'Set of 10 smooth-writing ballpoint pens in assorted colors', 8.99, '🖊️'),
        ('Coffee Mug', 'Home Office', 'Large ceramic coffee mug with funny programmer quote', 14.99, '☕'),
        ('Monitor Stand', 'Accessories', 'Wooden monitor riser with storage drawer', 44.99, '🖥️'),
        ('Wireless Earbuds', 'Electronics', 'Bluetooth 5.0 earbuds with noise cancellation', 69.99, '🎧'),
        ('Mouse Pad XL', 'Accessories', 'Extended gaming mouse pad with stitched edges', 19.99, '🎮'),
        ('Phone Holder', 'Accessories', 'Adjustable desk phone holder and tablet stand', 15.99, '📱'),
        ('Sticky Notes', 'Stationery', 'Pack of 12 colorful sticky note pads', 6.99, '📝'),
        ('Desk Organizer', 'Home Office', 'Bamboo desk organizer with multiple compartments', 27.99, '🗂️'),
        ('Blue Light Glasses', 'Accessories', 'Computer glasses that filter blue light for eye protection', 24.99, '👓'),
        ('Portable Charger', 'Electronics', '10000mAh portable power bank with fast charging', 35.99, '🔋'),
        ('Wrist Rest', 'Accessories', 'Memory foam keyboard wrist rest for comfort typing', 16.99, '🤲'),
        ('Whiteboard', 'Home Office', 'Magnetic dry erase whiteboard 24x36 inches', 29.99, '📋'),
        ('Tape Dispenser', 'Stationery', 'Weighted tape dispenser with non-slip base', 9.99, '🏷️'),
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
    <title>Simple Online Shop</title>
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
        .search-form {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        .search-input {
            padding: 12px 20px;
            font-size: 16px;
            border: none;
            border-radius: 25px;
            width: 350px;
            outline: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .search-input:focus {
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .search-btn {
            padding: 12px 25px;
            font-size: 16px;
            background-color: #ff6b6b;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .search-btn:hover {
            background-color: #ee5a5a;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .categories {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        .category-btn {
            padding: 8px 18px;
            background-color: white;
            border: 2px solid #667eea;
            color: #667eea;
            border-radius: 20px;
            text-decoration: none;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .category-btn:hover, .category-btn.active {
            background-color: #667eea;
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
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
        }
        .product-card {
            background-color: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.3s, box-shadow 0.3s;
            display: flex;
            flex-direction: column;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .product-emoji {
            font-size: 48px;
            text-align: center;
            margin-bottom: 15px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
        }
        .product-category {
            display: inline-block;
            padding: 4px 12px;
            background-color: #e8eaf6;
            color: #667eea;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 10px;
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
            line-height: 1.5;
            margin-bottom: 15px;
            flex-grow: 1;
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
        .add-to-cart-btn {
            padding: 8px 18px;
            background-color: #ff6b6b;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .add-to-cart-btn:hover {
            background-color: #ee5a5a;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .no-results h2 {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .no-results p {
            font-size: 16px;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #999;
            font-size: 14px;
            margin-top: 40px;
        }
        @media (max-width: 600px) {
            .search-input {
                width: 100%;
            }
            .header-content {
                flex-direction: column;
                text-align: center;
            }
            .search-form {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <a href="/" class="logo">🛍️ SimpleShop</a>
            <form class="search-form" action="/search" method="GET">
                <input 
                    type="text" 
                    name="q" 
                    class="search-input" 
                    placeholder="Search products by name, category, or keyword..." 
                    value="{{ query or '' }}"
                    autocomplete="off"
                >
                <button type="submit" class="search-btn">🔍 Search</button>
            </form>
        </div>
    </header>

    <div class="container">
        {% block content %}{% endblock %}
    </div>

    <footer>
        <p>&copy; 2024 SimpleShop — A beginner-friendly Python web shop demo</p>
    </footer>
</body>
</html>
'''

HOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <div class="categories">
        <a href="/" class="category-btn active">All Products</a>
        {% for cat in categories %}
        <a href="/search?q={{ cat }}" class="category-btn">{{ cat }}</a>
        {% endfor %}
    </div>

    <div class="results-info">
        Showing <strong>{{ products|length }}</strong> products
    </div>

    {% if products %}
    <div class="product-grid">
        {% for product in products %}
        <div class="product-card">
            <div class="product-emoji">{{ product.image_url }}</div>
            <span class="product-category">{{ product.category }}</span>
            <div class="product-name">{{ product.name }}</div>
            <div class="product-description">{{ product.description }}</div>
            <div class="product-footer">
                <span class="product-price">${{ "%.2f"|format(product.price) }}</span>
                <button class="add-to-cart-btn" onclick="alert('{{ product.name }} added to cart!')">Add to Cart</button>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="no-results">
        <h2>😕 No products found</h2>
        <p>Try a different search term or browse our categories above.</p>
    </div>
    {% endif %}
{% endblock %}
'''

SEARCH_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <div class="categories">
        <a href="/" class="category-btn">All Products</a>
        {% for cat in categories %}
        <a href="/search?q={{ cat }}" class="category-btn {% if query == cat %}active{% endif %}">{{ cat }}</a>
        {% endfor %}
    </div>

    <div class="results-info">
        {% if query %}
        Found <strong>{{ products|length }}</strong> result(s) for "<strong>{{ query }}</strong>"
        {% else %}
        Showing <strong>{{ products|length }}</strong> products
        {% endif %}
    </div>

    {% if products %}
    <div class="product-grid">
        {% for product in products %}
        <div class="product-card">
            <div class="product-emoji">{{ product.image_url }}</div>
            <span class="product-category">{{ product.category }}</span>
            <div class="product-name">{{ product.name }}</div>
            <div class="product-description">{{ product.description }}</div>
            <div class="product-footer">
                <span class="product-price">${{ "%.2f"|format(product.price) }}</span>
                <button class="add-to-cart-btn" onclick="alert('{{ product.name }} added to cart!')">Add to Cart</button>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="no-results">
        <h2>😕 No
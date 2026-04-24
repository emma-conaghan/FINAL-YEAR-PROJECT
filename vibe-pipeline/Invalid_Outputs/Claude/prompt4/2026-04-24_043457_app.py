from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)
DB_NAME = "shop.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT,
        price REAL NOT NULL
    )''')
    c.execute("SELECT COUNT(*) FROM products")
    count = c.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with A15 chip", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Flagship Android smartphone", 849.99),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99),
            ("Nike Air Max 270", "Shoes", "Comfortable running shoes with air cushion", 150.00),
            ("Adidas Ultraboost 22", "Shoes", "High performance running shoes", 180.00),
            ("Levi's 501 Jeans", "Clothing", "Classic straight fit denim jeans", 59.99),
            ("The North Face Jacket", "Clothing", "Warm winter jacket for outdoor activities", 249.99),
            ("Harry Potter Box Set", "Books", "Complete Harry Potter book series", 89.99),
            ("Python Programming Book", "Books", "Learn Python programming from scratch", 39.99),
            ("Coffee Maker Deluxe", "Kitchen", "Automatic drip coffee maker with timer", 79.99),
            ("Instant Pot 7-in-1", "Kitchen", "Multi-use pressure cooker and slow cooker", 99.99),
            ("Yoga Mat Pro", "Sports", "Non-slip premium yoga mat", 45.00),
            ("Dumbbells Set 20kg", "Sports", "Adjustable dumbbell set for home gym", 120.00),
            ("Leather Wallet", "Accessories", "Slim genuine leather bifold wallet", 29.99),
            ("Sunglasses UV400", "Accessories", "Polarized sunglasses with UV protection", 49.99),
            ("Bluetooth Speaker JBL", "Electronics", "Portable waterproof bluetooth speaker", 129.99),
            ("Gaming Mouse Logitech", "Electronics", "High precision wireless gaming mouse", 79.99),
            ("Desk Lamp LED", "Home", "Adjustable LED desk lamp with USB charging", 35.99),
            ("Cotton T-Shirt Pack", "Clothing", "Pack of 3 plain cotton t-shirts", 24.99),
            ("Running Water Bottle", "Sports", "BPA-free insulated sports water bottle", 19.99),
        ]
        c.executemany("INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)", sample_products)
    conn.commit()
    conn.close()

def search_products(query, category_filter):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if query and category_filter:
        like_query = f"%{query}%"
        c.execute(
            "SELECT id, name, category, description, price FROM products WHERE category = ? AND (name LIKE ? OR description LIKE ?)",
            (category_filter, like_query, like_query)
        )
    elif query:
        like_query = f"%{query}%"
        c.execute(
            "SELECT id, name, category, description, price FROM products WHERE name LIKE ? OR description LIKE ? OR category LIKE ?",
            (like_query, like_query, like_query)
        )
    elif category_filter:
        c.execute(
            "SELECT id, name, category, description, price FROM products WHERE category = ?",
            (category_filter,)
        )
    else:
        c.execute("SELECT id, name, category, description, price FROM products")
    results = c.fetchall()
    conn.close()
    return results

def get_categories():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT category FROM products ORDER BY category")
    categories = [row[0] for row in c.fetchall()]
    conn.close()
    return categories

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Arial, sans-serif;
            background: #f0f2f5;
            color: #333;
        }
        header {
            background: #2c3e50;
            color: white;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 26px; }
        header span { font-size: 14px; color: #bdc3c7; }
        .search-section {
            background: white;
            padding: 40px 20px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }
        .search-section h2 {
            font-size: 22px;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        form {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            max-width: 700px;
            margin: 0 auto;
        }
        input[type="text"] {
            flex: 1;
            min-width: 200px;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 15px;
            outline: none;
            transition: border 0.2s;
        }
        input[type="text"]:focus { border-color: #2980b9; }
        select {
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 15px;
            background: white;
            outline: none;
            cursor: pointer;
        }
        button {
            padding: 12px 24px;
            background: #2980b9;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #1a6fa3; }
        .featured {
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .featured h3 {
            font-size: 20px;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }
        .product-card .category {
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .product-card .name {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .product-card .desc {
            font-size: 13px;
            color: #7f8c8d;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 18px;
            font-weight: bold;
            color: #27ae60;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 13px;
            margin-top: 40px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Mini Online Shop</h1>
        <span>Find what you need, fast</span>
    </header>
    <div class="search-section">
        <h2>Search for Products</h2>
        <form action="/search" method="GET">
            <input type="text" name="q" placeholder="Search by name, keyword, or description..." />
            <select name="category">
                <option value="">All Categories</option>
                {% for cat in categories %}
                <option value="{{ cat }}">{{ cat }}</option>
                {% endfor %}
            </select>
            <button type="submit">🔍 Search</button>
        </form>
    </div>
    <div class="featured">
        <h3>All Products ({{ products|length }})</h3>
        <div class="product-grid">
            {% for p in products %}
            <div class="product-card">
                <div class="category">{{ p[2] }}</div>
                <div class="name">{{ p[1] }}</div>
                <div class="desc">{{ p[3] }}</div>
                <div class="price">${{ "%.2f"|format(p[4]) }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
    <footer>
        &copy; 2024 Mini Online Shop &mdash; Built with Flask & SQLite
    </footer>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results - Mini Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Arial, sans-serif;
            background: #f0f2f5;
            color: #333;
        }
        header {
            background: #2c3e50;
            color: white;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 26px; }
        header a {
            color: #bdc3c7;
            text-decoration: none;
            font-size: 14px;
        }
        header a:hover { color: white; }
        .search-bar {
            background: white;
            padding: 20px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: center;
        }
        form {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            max-width: 700px;
            width: 100%;
        }
        input[type="text"] {
            flex: 1;
            min-width: 200px;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 15px;
            outline: none;
            transition: border 0.2s;
        }
        input[type="text"]:focus { border-color: #2980b9; }
        select {
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 15px;
            background: white;
            outline: none;
            cursor: pointer;
        }
        button {
            padding: 12px 24px;
            background: #2980b9;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #1a6fa3; }
        .results-section {
            max-width: 1100px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .results-header {
            margin-bottom: 20px;
        }
        .results-header h2 {
            font-size: 20px;
            color: #2c3e50;
        }
        .results-header p {
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 4px;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }
        .product-card .category {
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .product-card .name {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .product-card .desc {
            font-size: 13px;
            color: #7f8c8d;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 18px;
            font-weight: bold;
            color: #27ae60;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #7f8c8d;
        }
        .no-results .icon { font-size: 50px; margin-bottom: 16px; }
        .no-results h3 { font-size: 22px; margin-bottom: 8px; }
        .no-results p { font-size: 15px; }
        footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 13px;
            margin-top: 40px;
            border-top: 1px solid #ddd
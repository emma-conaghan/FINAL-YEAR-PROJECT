from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)

DB_NAME = "shop.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with A15 chip", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Android smartphone with great camera", 849.99),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99),
            ("Nike Air Max 270", "Shoes", "Comfortable running shoes with air cushion", 150.00),
            ("Adidas Ultraboost 22", "Shoes", "High performance running shoes", 180.00),
            ("Levi's 501 Jeans", "Clothing", "Classic straight fit denim jeans", 59.99),
            ("The North Face Jacket", "Clothing", "Waterproof winter jacket", 220.00),
            ("Coffee Maker Deluxe", "Kitchen", "Programmable drip coffee maker", 79.99),
            ("Instant Pot Duo", "Kitchen", "7-in-1 electric pressure cooker", 99.99),
            ("Yoga Mat Premium", "Sports", "Non-slip exercise yoga mat", 35.00),
            ("Dumbbell Set 20kg", "Sports", "Adjustable dumbbell set for home gym", 120.00),
            ("Harry Potter Box Set", "Books", "Complete Harry Potter book collection", 85.00),
            ("Python Programming Book", "Books", "Learn Python programming from scratch", 45.00),
            ("Desk Lamp LED", "Home", "Adjustable LED desk lamp with USB port", 29.99),
            ("Bluetooth Speaker", "Electronics", "Portable waterproof bluetooth speaker", 59.99),
            ("Running Socks Pack", "Clothing", "Pack of 6 athletic running socks", 19.99),
            ("Blender Pro 900W", "Kitchen", "High speed blender for smoothies", 65.00),
            ("Football Official Size", "Sports", "FIFA approved match football", 40.00),
            ("Mystery Novel Bestseller", "Books", "Thrilling mystery novel top rated", 15.99),
            ("Scented Candle Set", "Home", "Set of 3 luxury scented candles", 24.99),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)",
            sample_products
        )
    conn.commit()
    conn.close()

def search_products(query, category_filter):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if category_filter and category_filter != "All":
        sql = '''
            SELECT id, name, category, description, price FROM products
            WHERE category = ?
            AND (name LIKE ? OR description LIKE ? OR category LIKE ?)
        '''
        like_query = f"%{query}%"
        cursor.execute(sql, (category_filter, like_query, like_query, like_query))
    else:
        sql = '''
            SELECT id, name, category, description, price FROM products
            WHERE name LIKE ? OR description LIKE ? OR category LIKE ?
        '''
        like_query = f"%{query}%"
        cursor.execute(sql, (like_query, like_query, like_query))
    results = cursor.fetchall()
    conn.close()
    return results

def get_categories():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM products ORDER BY category")
    cats = [row[0] for row in cursor.fetchall()]
    conn.close()
    return cats

HOME_TEMPLATE = '''
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
            background: #f0f4f8;
            min-height: 100vh;
        }
        header {
            background: #2c3e50;
            color: white;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 1.8rem; }
        header span { font-size: 0.9rem; color: #bdc3c7; }
        .hero {
            background: linear-gradient(135deg, #3498db, #8e44ad);
            color: white;
            text-align: center;
            padding: 60px 20px;
        }
        .hero h2 { font-size: 2.2rem; margin-bottom: 10px; }
        .hero p { font-size: 1.1rem; margin-bottom: 30px; color: #ecf0f1; }
        .search-box {
            background: white;
            border-radius: 12px;
            padding: 30px;
            max-width: 600px;
            margin: 0 auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .search-box form { display: flex; flex-direction: column; gap: 15px; }
        .search-box input[type="text"] {
            padding: 14px 18px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        .search-box input[type="text"]:focus { border-color: #3498db; }
        .search-box select {
            padding: 12px 18px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            outline: none;
            background: white;
            cursor: pointer;
        }
        .search-box button {
            padding: 14px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: background 0.3s;
        }
        .search-box button:hover { background: #2980b9; }
        .categories {
            max-width: 1000px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .categories h3 {
            font-size: 1.3rem;
            color: #2c3e50;
            margin-bottom: 15px;
            text-align: center;
        }
        .cat-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }
        .cat-link {
            display: inline-block;
            padding: 10px 20px;
            background: white;
            border: 2px solid #3498db;
            color: #3498db;
            border-radius: 25px;
            text-decoration: none;
            font-size: 0.95rem;
            transition: all 0.3s;
        }
        .cat-link:hover {
            background: #3498db;
            color: white;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.85rem;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Mini Shop</h1>
        <span>Find great products at great prices</span>
    </header>

    <div class="hero">
        <h2>Welcome to Mini Online Shop</h2>
        <p>Search thousands of products by name, category, or keyword</p>
        <div class="search-box">
            <form action="/search" method="GET">
                <input type="text" name="query" placeholder="Search for products... e.g. headphones, shoes, books" required>
                <select name="category">
                    <option value="All">All Categories</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}">{{ cat }}</option>
                    {% endfor %}
                </select>
                <button type="submit">🔍 Search Products</button>
            </form>
        </div>
    </div>

    <div class="categories">
        <h3>Browse by Category</h3>
        <div class="cat-grid">
            {% for cat in categories %}
            <a href="/search?query=&category={{ cat }}" class="cat-link">{{ cat }}</a>
            {% endfor %}
        </div>
    </div>

    <footer>
        <p>&copy; 2024 Mini Online Shop. Built with Flask &amp; SQLite.</p>
    </footer>
</body>
</html>
'''

RESULTS_TEMPLATE = '''
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
            background: #f0f4f8;
            min-height: 100vh;
        }
        header {
            background: #2c3e50;
            color: white;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 1.8rem; }
        header a {
            color: #3498db;
            text-decoration: none;
            font-size: 0.95rem;
        }
        header a:hover { text-decoration: underline; }
        .search-bar {
            background: #34495e;
            padding: 15px 40px;
        }
        .search-bar form {
            display: flex;
            gap: 10px;
            max-width: 800px;
            margin: 0 auto;
        }
        .search-bar input[type="text"] {
            flex: 1;
            padding: 10px 15px;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            outline: none;
        }
        .search-bar select {
            padding: 10px 15px;
            border: none;
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            background: white;
            cursor: pointer;
        }
        .search-bar button {
            padding: 10px 20px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
        }
        .search-bar button:hover { background: #2980b9; }
        .results-info {
            max-width: 1100px;
            margin: 25px auto 15px;
            padding: 0 20px;
            color: #555;
            font-size: 1rem;
        }
        .results-info span { color: #2c3e50; font-weight: bold; }
        .results-grid {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 20px 40px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }
        .product-icon {
            font-size: 2.5rem;
            text-align: center;
        }
        .product-name {
            font-size: 1.05rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .product-category {
            display: inline-block;
            background: #eaf4fb;
            color: #2980b9;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .product-description {
            color: #666;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        .product-price {
            font-size: 1.3rem;
            font-weight: bold;
            color: #27ae60;
            margin-top: auto;
        }
        .add-btn {
            padding: 10px;
            background: #27ae60;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 0.95rem;
            cursor: pointer;
            transition: background 0.3s;
        }
        .add-btn:hover { background: #219a52; }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #7f8c8d;
        }
        .no-results h3 { font-size: 1.5rem; margin-bottom: 10px; }
        .no-results a {
            color: #3498db;
            text-decoration: none;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.85rem;
            margin-top: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Mini Shop</h1>
        <a href="/">← Back to Home</a>
    </header>

    <div class="search-bar">
        <form action="/search" method="GET">
            <input type="text" name="query" value="{{ query }}" placeholder="Search products...">
            <select name="category">
                <option value="All">All Categories</option>
                {% for cat in categories %}
                <option value="{{ cat }}" {% if cat == selected_category %}selected{% endif %}>{{ cat
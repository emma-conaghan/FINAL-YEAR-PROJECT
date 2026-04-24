from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)

DB_NAME = "shop.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with A15 chip", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Android flagship smartphone", 849.99),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99),
            ("Nike Air Max 270", "Shoes", "Comfortable running shoes with air cushion", 150.00),
            ("Adidas Ultraboost 22", "Shoes", "High performance running shoes", 180.00),
            ("Levi's 501 Jeans", "Clothing", "Classic straight fit jeans", 69.99),
            ("H&M Cotton T-Shirt", "Clothing", "Basic everyday cotton t-shirt", 19.99),
            ("The Great Gatsby", "Books", "Classic novel by F. Scott Fitzgerald", 12.99),
            ("Python Crash Course", "Books", "Beginner-friendly Python programming book", 35.00),
            ("LEGO Technic Car", "Toys", "Advanced building set for ages 10+", 59.99),
            ("Barbie Dreamhouse", "Toys", "Large dollhouse with accessories", 199.99),
            ("Instant Pot Duo", "Kitchen", "7-in-1 electric pressure cooker", 89.99),
            ("KitchenAid Stand Mixer", "Kitchen", "Professional stand mixer for baking", 449.99),
            ("Yoga Mat Premium", "Sports", "Non-slip exercise mat for yoga and fitness", 39.99),
            ("Dumbbells Set 20kg", "Sports", "Adjustable dumbbell set for home gym", 75.00),
            ("Coffee Table Oak", "Furniture", "Solid oak wood coffee table", 299.99),
            ("Office Chair Ergonomic", "Furniture", "Adjustable ergonomic desk chair", 249.99),
            ("Moisturizing Face Cream", "Beauty", "Daily hydrating face cream with SPF", 24.99),
            ("Vitamin C Serum", "Beauty", "Brightening vitamin C face serum", 18.99),
            ("Dog Food Premium", "Pet Supplies", "Grain-free dry food for adult dogs", 45.00),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)",
            sample_products
        )
    conn.commit()
    conn.close()

def search_products(query, category_filter=""):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if category_filter and query:
        cursor.execute("""
            SELECT id, name, category, description, price FROM products
            WHERE category = ? AND (
                name LIKE ? OR description LIKE ? OR category LIKE ?
            )
        """, (category_filter, f"%{query}%", f"%{query}%", f"%{query}%"))
    elif category_filter:
        cursor.execute("""
            SELECT id, name, category, description, price FROM products
            WHERE category = ?
        """, (category_filter,))
    elif query:
        cursor.execute("""
            SELECT id, name, category, description, price FROM products
            WHERE name LIKE ? OR description LIKE ? OR category LIKE ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%"))
    else:
        cursor.execute("SELECT id, name, category, description, price FROM products")
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

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Online Shop - Search Products</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 40px;
            text-align: center;
        }
        header h1 {
            font-size: 2em;
        }
        header p {
            font-size: 1em;
            color: #bdc3c7;
            margin-top: 5px;
        }
        nav {
            background-color: #34495e;
            padding: 10px 40px;
            display: flex;
            gap: 20px;
        }
        nav a {
            color: #ecf0f1;
            text-decoration: none;
            font-size: 0.95em;
        }
        nav a:hover {
            color: #3498db;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .search-box {
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .search-box h2 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .search-form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .form-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-input {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            min-width: 200px;
        }
        .search-input:focus {
            outline: none;
            border-color: #3498db;
        }
        .category-select {
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            background: white;
            cursor: pointer;
        }
        .category-select:focus {
            outline: none;
            border-color: #3498db;
        }
        .search-btn {
            padding: 12px 30px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.2s;
        }
        .search-btn:hover {
            background-color: #2980b9;
        }
        .tips {
            margin-top: 20px;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
            font-size: 0.9em;
            color: #555;
        }
        .tips strong {
            display: block;
            margin-bottom: 5px;
            color: #2c3e50;
        }
        .category-cards {
            margin-top: 40px;
        }
        .category-cards h2 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 15px;
        }
        .cat-card {
            background: white;
            border-radius: 8px;
            padding: 20px 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-decoration: none;
            color: #2c3e50;
            font-size: 0.95em;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .cat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            color: #3498db;
        }
        .cat-icon {
            font-size: 2em;
            margin-bottom: 8px;
        }
        footer {
            text-align: center;
            padding: 20px;
            margin-top: 60px;
            color: #888;
            font-size: 0.85em;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <p>Find the best products at great prices</p>
    </header>
    <nav>
        <a href="/">🏠 Home</a>
        <a href="/search">🔍 All Products</a>
    </nav>
    <div class="container">
        <div class="search-box">
            <h2>🔍 Search Products</h2>
            <form class="search-form" action="/search" method="get">
                <div class="form-row">
                    <input
                        class="search-input"
                        type="text"
                        name="q"
                        placeholder="Search by name, keyword, or description..."
                        autofocus
                    >
                    <select class="category-select" name="category">
                        <option value="">All Categories</option>
                        {% for cat in categories %}
                        <option value="{{ cat }}">{{ cat }}</option>
                        {% endfor %}
                    </select>
                    <button class="search-btn" type="submit">Search</button>
                </div>
            </form>
            <div class="tips">
                <strong>💡 Search Tips:</strong>
                Try searching for "headphones", "running", "kitchen", or browse by category above.
            </div>
        </div>

        <div class="category-cards">
            <h2>📦 Browse by Category</h2>
            <div class="cards-grid">
                {% set icons = {
                    'Electronics': '📱',
                    'Shoes': '👟',
                    'Clothing': '👕',
                    'Books': '📚',
                    'Toys': '🧸',
                    'Kitchen': '🍳',
                    'Sports': '⚽',
                    'Furniture': '🪑',
                    'Beauty': '💄',
                    'Pet Supplies': '🐾'
                } %}
                {% for cat in categories %}
                <a class="cat-card" href="/search?category={{ cat }}">
                    <div class="cat-icon">{{ icons.get(cat, '🛍️') }}</div>
                    {{ cat }}
                </a>
                {% endfor %}
            </div>
        </div>
    </div>
    <footer>
        &copy; 2024 My Online Shop. Built with Flask &amp; SQLite.
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
    <title>Search Results - Online Shop</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 40px;
            text-align: center;
        }
        header h1 {
            font-size: 2em;
        }
        header p {
            font-size: 1em;
            color: #bdc3c7;
            margin-top: 5px;
        }
        nav {
            background-color: #34495e;
            padding: 10px 40px;
            display: flex;
            gap: 20px;
        }
        nav a {
            color: #ecf0f1;
            text-decoration: none;
            font-size: 0.95em;
        }
        nav a:hover {
            color: #3498db;
        }
        .container {
            max-width: 960px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .search-bar {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }
        .search-form {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-input {
            flex: 1;
            padding: 10px 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            min-width: 200px;
        }
        .search-input:focus {
            outline: none;
            border-color: #3498db;
        }
        .category-select {
            padding: 10px 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            background: white;
            cursor: pointer;
        }
        .search-btn {
            padding: 10px 25px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
        }
        .search-btn:hover {
            background-color: #2980b9;
        }
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .results-count {
            font-size: 1em;
            color: #555;
        }
        .results-count strong {
            color: #2c3e50;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10
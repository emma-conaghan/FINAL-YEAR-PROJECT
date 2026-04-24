from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)

DB_PATH = "shop.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            keyword TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with A15 chip", 999.99, "phone mobile apple"),
            ("Samsung Galaxy S23", "Electronics", "Flagship Samsung Android phone", 899.99, "phone mobile samsung android"),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99, "audio music headphones wireless"),
            ("Nike Air Max 270", "Shoes", "Comfortable running shoes with air cushion", 149.99, "running sport shoes nike"),
            ("Adidas Ultraboost 22", "Shoes", "High performance running sneakers", 179.99, "running sport shoes adidas"),
            ("Levi's 501 Jeans", "Clothing", "Classic straight fit denim jeans", 59.99, "denim pants fashion jeans"),
            ("The Great Gatsby", "Books", "Classic novel by F. Scott Fitzgerald", 12.99, "novel fiction literature classic"),
            ("Python Programming for Beginners", "Books", "Learn Python programming from scratch", 29.99, "coding programming tech education"),
            ("Wooden Coffee Table", "Furniture", "Minimalist wooden coffee table for living room", 199.99, "table wood living room home"),
            ("Office Chair", "Furniture", "Ergonomic office chair with lumbar support", 249.99, "chair desk work ergonomic office"),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for exercise", 39.99, "yoga fitness exercise sport mat"),
            ("Dumbbells Set", "Sports", "Adjustable dumbbell set for home gym", 89.99, "weights gym fitness exercise"),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker with timer", 79.99, "coffee kitchen appliance morning"),
            ("Air Fryer", "Kitchen", "Digital air fryer for healthy cooking", 119.99, "cooking kitchen healthy appliance"),
            ("Backpack", "Bags", "Waterproof travel backpack with laptop compartment", 69.99, "travel bag laptop school"),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, description, price, keyword) VALUES (?, ?, ?, ?, ?)",
            sample_products
        )
    conn.commit()
    conn.close()

def search_products(query):
    if not query or query.strip() == "":
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    search_term = f"%{query.strip()}%"
    cursor.execute("""
        SELECT id, name, category, description, price, keyword
        FROM products
        WHERE name LIKE ?
           OR category LIKE ?
           OR description LIKE ?
           OR keyword LIKE ?
        ORDER BY name ASC
    """, (search_term, search_term, search_term, search_term))
    results = cursor.fetchall()
    conn.close()
    return results

def get_all_products():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, category, description, price, keyword FROM products ORDER BY category, name")
    results = cursor.fetchall()
    conn.close()
    return results

def get_categories():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM products ORDER BY category")
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
    return categories

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Small Online Shop</title>
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
            padding: 20px;
            text-align: center;
        }
        header h1 {
            font-size: 2em;
            margin-bottom: 5px;
        }
        header p {
            font-size: 1em;
            color: #bdc3c7;
        }
        nav {
            background-color: #34495e;
            padding: 10px;
            text-align: center;
        }
        nav a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
            font-size: 1em;
        }
        nav a:hover {
            text-decoration: underline;
        }
        .search-section {
            max-width: 700px;
            margin: 50px auto;
            text-align: center;
            padding: 0 20px;
        }
        .search-section h2 {
            font-size: 1.6em;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .search-form {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            flex: 1;
            min-width: 250px;
            padding: 12px 16px;
            font-size: 1em;
            border: 2px solid #ccc;
            border-radius: 6px;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #2980b9;
        }
        .search-form button {
            padding: 12px 24px;
            font-size: 1em;
            background-color: #2980b9;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .search-form button:hover {
            background-color: #1a6fa3;
        }
        .categories-section {
            max-width: 900px;
            margin: 30px auto;
            padding: 0 20px;
            text-align: center;
        }
        .categories-section h3 {
            margin-bottom: 15px;
            color: #555;
            font-size: 1.1em;
        }
        .category-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }
        .category-tag {
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            border-radius: 20px;
            padding: 6px 16px;
            text-decoration: none;
            color: #2c3e50;
            font-size: 0.9em;
        }
        .category-tag:hover {
            background-color: #2980b9;
            color: white;
            border-color: #2980b9;
        }
        .all-products-link {
            display: inline-block;
            margin-top: 30px;
            padding: 10px 20px;
            background-color: #27ae60;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 1em;
        }
        .all-products-link:hover {
            background-color: #1e8449;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.85em;
            margin-top: 60px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Small Online Shop</h1>
        <p>Find exactly what you need</p>
    </header>
    <nav>
        <a href="/">Home</a>
        <a href="/products">All Products</a>
    </nav>
    <div class="search-section">
        <h2>Search for Products</h2>
        <form class="search-form" action="/search" method="GET">
            <input type="text" name="q" placeholder="Search by name, category, or keyword..." autofocus>
            <button type="submit">Search</button>
        </form>
    </div>
    <div class="categories-section">
        <h3>Browse by Category</h3>
        <div class="category-tags">
            {% for category in categories %}
            <a class="category-tag" href="/search?q={{ category }}">{{ category }}</a>
            {% endfor %}
        </div>
        <br>
        <a class="all-products-link" href="/products">View All Products</a>
    </div>
    <footer>
        <p>&copy; 2024 Small Online Shop. All rights reserved.</p>
    </footer>
</body>
</html>
"""

SEARCH_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results - Small Online Shop</title>
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
            padding: 20px;
            text-align: center;
        }
        header h1 {
            font-size: 2em;
            margin-bottom: 5px;
        }
        nav {
            background-color: #34495e;
            padding: 10px;
            text-align: center;
        }
        nav a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
            font-size: 1em;
        }
        nav a:hover {
            text-decoration: underline;
        }
        .search-bar {
            max-width: 700px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .search-form {
            display: flex;
            gap: 10px;
        }
        .search-form input[type="text"] {
            flex: 1;
            padding: 12px 16px;
            font-size: 1em;
            border: 2px solid #ccc;
            border-radius: 6px;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #2980b9;
        }
        .search-form button {
            padding: 12px 24px;
            font-size: 1em;
            background-color: #2980b9;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .search-form button:hover {
            background-color: #1a6fa3;
        }
        .results-section {
            max-width: 1000px;
            margin: 0 auto 40px auto;
            padding: 0 20px;
        }
        .results-header {
            margin-bottom: 20px;
        }
        .results-header h2 {
            font-size: 1.4em;
            color: #2c3e50;
        }
        .results-header p {
            color: #777;
            margin-top: 5px;
            font-size: 0.95em;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #888;
            font-size: 1.1em;
        }
        .no-results a {
            color: #2980b9;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
        }
        .product-card {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            transition: box-shadow 0.2s;
        }
        .product-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
        .product-name {
            font-size: 1.1em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 6px;
        }
        .product-category {
            display: inline-block;
            background-color: #eaf4fb;
            color: #2980b9;
            border-radius: 12px;
            padding: 3px 10px;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .product-description {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-price {
            font-size: 1.2em;
            font-weight: bold;
            color: #27ae60;
        }
        .back-link {
            display: inline-block;
            margin: 20px 0;
            color: #2980b9;
            text-decoration: none;
            font-size: 0.95em;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.85em;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Small Online Shop</h1>
    </header>
    <nav>
        <a href="/">Home</a>
        <a href="/products">All Products</a>
    </nav>
    <div class="search-bar">
        <form class="search-form" action="/search" method="GET">
            <input type="text" name="q" value="{{ query }}" placeholder="Search by name, category, or keyword...">
            <button type="submit">Search</button>
        </form>
    </div>
    <div class="results-section">
        <a class="back-link" href="/">← Back to Home</a>
        <div class="results-header">
            <h
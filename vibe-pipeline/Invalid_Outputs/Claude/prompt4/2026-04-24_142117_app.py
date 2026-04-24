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
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with great camera", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Android smartphone with high refresh rate screen", 849.99),
            ("Nike Air Max", "Shoes", "Comfortable running shoes for everyday use", 129.99),
            ("Adidas Ultraboost", "Shoes", "High performance running shoes with boost technology", 179.99),
            ("Python Programming Book", "Books", "Learn Python from scratch with hands-on examples", 39.99),
            ("JavaScript Guide", "Books", "Complete guide to modern JavaScript development", 34.99),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker with timer", 59.99),
            ("Blender Pro", "Kitchen", "High speed blender for smoothies and soups", 89.99),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for home workouts", 29.99),
            ("Dumbbells Set", "Sports", "Adjustable dumbbells set for strength training", 149.99),
            ("Laptop Stand", "Office", "Adjustable aluminum laptop stand for desk use", 49.99),
            ("Mechanical Keyboard", "Office", "Tactile mechanical keyboard with RGB lighting", 109.99),
            ("Sunglasses", "Accessories", "UV protection sunglasses with polarized lenses", 59.99),
            ("Leather Wallet", "Accessories", "Slim genuine leather wallet with card slots", 39.99),
            ("Wireless Headphones", "Electronics", "Noise cancelling bluetooth headphones", 199.99),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)",
            sample_products
        )
    conn.commit()
    conn.close()

def search_products(query):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    search_term = f"%{query}%"
    cursor.execute('''
        SELECT id, name, category, description, price
        FROM products
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
    ''', (search_term, search_term, search_term))
    results = cursor.fetchall()
    conn.close()
    return results

def get_all_products():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, category, description, price FROM products")
    results = cursor.fetchall()
    conn.close()
    return results

def get_categories():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM products ORDER BY category")
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
    return categories

HOME_TEMPLATE = '''
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
            background-color: #f4f4f4;
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
            padding: 10px 20px;
            text-align: center;
        }
        nav a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
            font-size: 1em;
        }
        nav a:hover {
            color: #3498db;
        }
        .search-section {
            background-color: white;
            padding: 40px 20px;
            text-align: center;
            margin: 20px auto;
            max-width: 700px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .search-section h2 {
            margin-bottom: 20px;
            font-size: 1.5em;
            color: #2c3e50;
        }
        .search-form {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            padding: 12px 16px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 6px;
            width: 350px;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #3498db;
        }
        .search-form button {
            padding: 12px 24px;
            font-size: 1em;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .search-form button:hover {
            background-color: #2980b9;
        }
        .categories-section {
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
        }
        .categories-section h3 {
            margin-bottom: 15px;
            font-size: 1.2em;
            color: #2c3e50;
        }
        .category-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .category-btn {
            padding: 8px 18px;
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            border-radius: 20px;
            text-decoration: none;
            color: #2c3e50;
            font-size: 0.9em;
            transition: background-color 0.2s;
        }
        .category-btn:hover {
            background-color: #3498db;
            color: white;
            border-color: #3498db;
        }
        .products-section {
            max-width: 1100px;
            margin: 20px auto;
            padding: 0 20px 40px 20px;
        }
        .products-section h3 {
            margin-bottom: 20px;
            font-size: 1.3em;
            color: #2c3e50;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background-color: white;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
        }
        .product-card .product-category {
            font-size: 0.75em;
            color: #7f8c8d;
            background-color: #ecf0f1;
            padding: 3px 8px;
            border-radius: 12px;
            display: inline-block;
            margin-bottom: 10px;
        }
        .product-card h4 {
            font-size: 1em;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .product-card p {
            font-size: 0.85em;
            color: #7f8c8d;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 1.2em;
            font-weight: bold;
            color: #27ae60;
        }
        footer {
            text-align: center;
            padding: 20px;
            background-color: #2c3e50;
            color: #bdc3c7;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Small Online Shop</h1>
        <p>Find the best products at great prices</p>
    </header>
    <nav>
        <a href="/">Home</a>
        <a href="/products">All Products</a>
    </nav>

    <div class="search-section">
        <h2>Search for Products</h2>
        <form class="search-form" action="/search" method="get">
            <input type="text" name="q" placeholder="Search by name, category, or keyword..." required>
            <button type="submit">🔍 Search</button>
        </form>
    </div>

    <div class="categories-section">
        <h3>Browse by Category</h3>
        <div class="category-buttons">
            {% for category in categories %}
            <a href="/search?q={{ category }}" class="category-btn">{{ category }}</a>
            {% endfor %}
        </div>
    </div>

    <div class="products-section">
        <h3>Featured Products</h3>
        <div class="products-grid">
            {% for product in products %}
            <div class="product-card">
                <span class="product-category">{{ product[2] }}</span>
                <h4>{{ product[1] }}</h4>
                <p>{{ product[3] }}</p>
                <span class="price">${{ "%.2f"|format(product[4]) }}</span>
            </div>
            {% endfor %}
        </div>
    </div>

    <footer>
        <p>&copy; 2024 Small Online Shop. All rights reserved.</p>
    </footer>
</body>
</html>
'''

SEARCH_TEMPLATE = '''
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
            background-color: #f4f4f4;
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
            padding: 10px 20px;
            text-align: center;
        }
        nav a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
            font-size: 1em;
        }
        nav a:hover {
            color: #3498db;
        }
        .search-section {
            background-color: white;
            padding: 25px 20px;
            text-align: center;
            margin: 20px auto;
            max-width: 700px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .search-form {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            padding: 12px 16px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 6px;
            width: 350px;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #3498db;
        }
        .search-form button {
            padding: 12px 24px;
            font-size: 1em;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .search-form button:hover {
            background-color: #2980b9;
        }
        .results-section {
            max-width: 1100px;
            margin: 20px auto;
            padding: 0 20px 40px 20px;
        }
        .results-header {
            margin-bottom: 20px;
        }
        .results-header h2 {
            font-size: 1.4em;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .results-header p {
            font-size: 0.95em;
            color: #7f8c8d;
        }
        .no-results {
            background-color: white;
            padding: 40px;
            text-align: center;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .no-results h3 {
            font-size: 1.2em;
            color: #e74c3c;
            margin-bottom: 10px;
        }
        .no-results p {
            color: #7f8c8d;
            margin-bottom: 15px;
        }
        .no-results a {
            color: #3498db;
            text-decoration: none;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background-color: white;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
        }
        .product-card .product-category {
            font-size: 0.75em;
            color: #7f8c8d;
            background-color: #ecf0f1;
            padding: 3px 8px;
            border-radius: 12px;
            display: inline
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
            ("Samsung Galaxy S23", "Electronics", "Flagship Android phone with great camera", 849.99),
            ("Nike Air Max", "Shoes", "Comfortable running shoes with air cushioning", 129.99),
            ("Adidas Ultraboost", "Shoes", "High performance running shoes", 179.99),
            ("Python Programming Book", "Books", "Learn Python from scratch with examples", 39.99),
            ("JavaScript for Beginners", "Books", "A beginner friendly guide to JavaScript", 34.99),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker for home use", 49.99),
            ("Blender Pro", "Kitchen", "High speed blender for smoothies and soups", 79.99),
            ("Yoga Mat", "Sports", "Non slip yoga mat for home workouts", 25.99),
            ("Dumbbells Set", "Sports", "Adjustable dumbbell set for strength training", 149.99),
            ("Laptop Stand", "Office", "Ergonomic laptop stand for better posture", 35.99),
            ("Wireless Mouse", "Office", "Compact wireless mouse with long battery life", 29.99),
            ("Headphones", "Electronics", "Noise cancelling over ear headphones", 199.99),
            ("Backpack", "Accessories", "Durable backpack with multiple compartments", 59.99),
            ("Water Bottle", "Sports", "Insulated stainless steel water bottle", 22.99),
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
    cursor.execute("""
        SELECT id, name, category, description, price
        FROM products
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        ORDER BY name ASC
    """, (search_term, search_term, search_term))
    results = cursor.fetchall()
    conn.close()
    return results

def get_all_products():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, category, description, price FROM products ORDER BY name ASC")
    results = cursor.fetchall()
    conn.close()
    return results

def get_categories():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM products ORDER BY category ASC")
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
            margin: 0;
            padding: 0;
            box-sizing: border-box;
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
        .search-section {
            background-color: white;
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .search-section h2 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .search-form {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            padding: 12px 20px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 25px;
            width: 350px;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #2c3e50;
        }
        .search-form button {
            padding: 12px 25px;
            font-size: 1em;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
        }
        .search-form button:hover {
            background-color: #c0392b;
        }
        .categories {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .category-btn {
            padding: 8px 16px;
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            text-decoration: none;
            color: #333;
        }
        .category-btn:hover {
            background-color: #2c3e50;
            color: white;
        }
        .products-section {
            max-width: 1100px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .products-section h2 {
            margin-bottom: 20px;
            color: #2c3e50;
            font-size: 1.5em;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
        }
        .product-card .category-tag {
            background-color: #2c3e50;
            color: white;
            font-size: 0.75em;
            padding: 3px 10px;
            border-radius: 12px;
            display: inline-block;
            margin-bottom: 10px;
        }
        .product-card h3 {
            font-size: 1em;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .product-card p {
            font-size: 0.85em;
            color: #777;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 1.2em;
            font-weight: bold;
            color: #e74c3c;
        }
        footer {
            text-align: center;
            padding: 20px;
            background-color: #2c3e50;
            color: #bdc3c7;
            margin-top: 40px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <p>Find great products at great prices</p>
    </header>

    <div class="search-section">
        <h2>Search for Products</h2>
        <form class="search-form" action="/search" method="GET">
            <input type="text" name="q" placeholder="Search by name, category, or keyword..." required>
            <button type="submit">Search</button>
        </form>
        <div class="categories">
            <span style="font-size:0.9em; color:#777; margin-top:5px;">Browse by category:</span>
            {% for cat in categories %}
            <a class="category-btn" href="/search?q={{ cat }}">{{ cat }}</a>
            {% endfor %}
        </div>
    </div>

    <div class="products-section">
        <h2>All Products ({{ products|length }})</h2>
        <div class="products-grid">
            {% for product in products %}
            <div class="product-card">
                <span class="category-tag">{{ product[2] }}</span>
                <h3>{{ product[1] }}</h3>
                <p>{{ product[3] }}</p>
                <div class="price">${{ "%.2f"|format(product[4]) }}</div>
            </div>
            {% endfor %}
        </div>
    </div>

    <footer>
        <p>&copy; 2024 My Online Shop. All rights reserved.</p>
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
            margin: 0;
            padding: 0;
            box-sizing: border-box;
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
        header a {
            color: #bdc3c7;
            text-decoration: none;
            font-size: 0.9em;
        }
        header a:hover {
            color: white;
        }
        .search-section {
            background-color: white;
            padding: 20px 30px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .search-form {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            padding: 12px 20px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 25px;
            width: 350px;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #2c3e50;
        }
        .search-form button {
            padding: 12px 25px;
            font-size: 1em;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
        }
        .search-form button:hover {
            background-color: #c0392b;
        }
        .results-section {
            max-width: 1100px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .results-header {
            margin-bottom: 20px;
        }
        .results-header h2 {
            color: #2c3e50;
            font-size: 1.4em;
        }
        .results-header p {
            color: #777;
            font-size: 0.95em;
            margin-top: 5px;
        }
        .no-results {
            text-align: center;
            padding: 50px 20px;
            color: #777;
        }
        .no-results h3 {
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        .no-results a {
            color: #e74c3c;
            text-decoration: none;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
        }
        .product-card .category-tag {
            background-color: #2c3e50;
            color: white;
            font-size: 0.75em;
            padding: 3px 10px;
            border-radius: 12px;
            display: inline-block;
            margin-bottom: 10px;
        }
        .product-card h3 {
            font-size: 1em;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .product-card p {
            font-size: 0.85em;
            color: #777;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 1.2em;
            font-weight: bold;
            color: #e74c3c;
        }
        .back-link {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background-color: #2c3e50;
            color: white;
            text-decoration: none;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .back-link:hover {
            background-color: #34495e;
        }
        footer {
            text-align: center;
            padding: 20px;
            background-color: #2c3e50;
            color: #bdc3c7;
            margin-top: 40px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <a href="/">← Back to Home</a>
    </header>

    <div class="search-section">
        <form class="search-form" action="/search
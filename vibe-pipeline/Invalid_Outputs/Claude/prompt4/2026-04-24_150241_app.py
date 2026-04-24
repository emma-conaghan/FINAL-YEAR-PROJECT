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
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with great camera", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Android flagship phone with stunning display", 849.99),
            ("Nike Air Max", "Shoes", "Comfortable running shoes with air cushioning", 129.99),
            ("Adidas Ultraboost", "Shoes", "High performance running shoes", 180.00),
            ("Python Programming Book", "Books", "Learn Python from scratch for beginners", 39.99),
            ("JavaScript Guide", "Books", "Complete guide to modern JavaScript", 34.99),
            ("Coffee Maker", "Kitchen", "Brew delicious coffee every morning", 59.99),
            ("Blender Pro", "Kitchen", "Powerful blender for smoothies and more", 79.99),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for all exercises", 25.00),
            ("Dumbbells Set", "Sports", "Adjustable dumbbell set for home workouts", 150.00),
            ("Wireless Headphones", "Electronics", "Noise cancelling wireless headphones", 299.99),
            ("Laptop Stand", "Electronics", "Ergonomic aluminum laptop stand", 45.00),
            ("Harry Potter Box Set", "Books", "Complete Harry Potter book collection", 89.99),
            ("Cooking Pan Set", "Kitchen", "Non-stick cooking pan set of 3", 65.00),
            ("Running Shorts", "Clothes", "Lightweight breathable running shorts", 22.99),
            ("Cotton T-Shirt", "Clothes", "Soft comfortable everyday cotton t-shirt", 15.99),
            ("Winter Jacket", "Clothes", "Warm insulated jacket for cold weather", 120.00),
            ("Basketball", "Sports", "Official size basketball for indoor and outdoor", 35.00),
            ("Water Bottle", "Sports", "Insulated stainless steel water bottle", 20.00),
            ("Electric Kettle", "Kitchen", "Fast boiling electric kettle with auto shutoff", 40.00),
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
    <title>Online Shop</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #f0f2f5;
            color: #333;
        }
        header {
            background: #2c3e50;
            color: white;
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 24px; }
        header a { color: #ecf0f1; text-decoration: none; font-size: 14px; }
        header a:hover { text-decoration: underline; }
        .hero {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            text-align: center;
            padding: 60px 20px;
        }
        .hero h2 { font-size: 36px; margin-bottom: 10px; }
        .hero p { font-size: 18px; margin-bottom: 30px; opacity: 0.9; }
        .search-form {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            padding: 12px 20px;
            font-size: 16px;
            border: none;
            border-radius: 25px;
            width: 400px;
            max-width: 90vw;
            outline: none;
        }
        .search-form button {
            padding: 12px 30px;
            font-size: 16px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
        }
        .search-form button:hover { background: #c0392b; }
        .categories {
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .categories h3 { font-size: 22px; margin-bottom: 15px; color: #2c3e50; }
        .category-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .category-btn {
            background: white;
            border: 2px solid #3498db;
            color: #3498db;
            padding: 8px 18px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 14px;
            transition: all 0.2s;
        }
        .category-btn:hover {
            background: #3498db;
            color: white;
        }
        .featured {
            max-width: 1100px;
            margin: 20px auto 40px;
            padding: 0 20px;
        }
        .featured h3 { font-size: 22px; margin-bottom: 15px; color: #2c3e50; }
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
        .product-card .category-tag {
            font-size: 12px;
            background: #eaf4fd;
            color: #2980b9;
            padding: 3px 10px;
            border-radius: 10px;
            display: inline-block;
            margin-bottom: 8px;
        }
        .product-card h4 { font-size: 16px; margin-bottom: 6px; }
        .product-card p { font-size: 13px; color: #777; margin-bottom: 10px; }
        .product-card .price { font-size: 18px; font-weight: bold; color: #e74c3c; }
        footer {
            background: #2c3e50;
            color: #ecf0f1;
            text-align: center;
            padding: 20px;
            font-size: 14px;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 MyShop</h1>
        <a href="/">Home</a>
    </header>
    <div class="hero">
        <h2>Find What You Need</h2>
        <p>Search thousands of products by name, category, or keyword</p>
        <form class="search-form" action="/search" method="get">
            <input type="text" name="q" placeholder="Search products..." required>
            <button type="submit">Search</button>
        </form>
    </div>
    <div class="categories">
        <h3>Browse by Category</h3>
        <div class="category-list">
            {% for cat in categories %}
            <a class="category-btn" href="/search?q={{ cat }}">{{ cat }}</a>
            {% endfor %}
        </div>
    </div>
    <div class="featured">
        <h3>All Products</h3>
        <div class="product-grid">
            {% for product in products %}
            <div class="product-card">
                <span class="category-tag">{{ product[2] }}</span>
                <h4>{{ product[1] }}</h4>
                <p>{{ product[3] }}</p>
                <div class="price">${{ "%.2f"|format(product[4]) }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
    <footer>
        <p>&copy; 2024 MyShop. All rights reserved.</p>
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
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #f0f2f5;
            color: #333;
        }
        header {
            background: #2c3e50;
            color: white;
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 24px; }
        header a { color: #ecf0f1; text-decoration: none; font-size: 14px; }
        header a:hover { text-decoration: underline; }
        .search-bar {
            background: #34495e;
            padding: 20px 30px;
        }
        .search-form {
            display: flex;
            gap: 10px;
            max-width: 700px;
            margin: 0 auto;
        }
        .search-form input[type="text"] {
            flex: 1;
            padding: 10px 18px;
            font-size: 15px;
            border: none;
            border-radius: 20px;
            outline: none;
        }
        .search-form button {
            padding: 10px 25px;
            font-size: 15px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
        }
        .search-form button:hover { background: #c0392b; }
        .results-info {
            max-width: 1100px;
            margin: 25px auto 10px;
            padding: 0 20px;
            font-size: 15px;
            color: #555;
        }
        .results-info span { font-weight: bold; color: #2c3e50; }
        .product-grid {
            max-width: 1100px;
            margin: 10px auto 40px;
            padding: 0 20px;
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
        .product-card .category-tag {
            font-size: 12px;
            background: #eaf4fd;
            color: #2980b9;
            padding: 3px 10px;
            border-radius: 10px;
            display: inline-block;
            margin-bottom: 8px;
        }
        .product-card h4 { font-size: 16px; margin-bottom: 6px; }
        .product-card p { font-size: 13px; color: #777; margin-bottom: 10px; }
        .product-card .price { font-size: 18px; font-weight: bold; color: #e74c3c; }
        .no-results {
            max-width: 600px;
            margin: 60px auto;
            text-align: center;
            padding: 0 20px;
        }
        .no-results h2 { font-size: 26px; color: #2c3e50; margin-bottom: 10px; }
        .no-results p { color: #777; margin-bottom: 20px; }
        .no-results a {
            display: inline-block;
            padding: 10px 25px;
            background: #3498db;
            color: white;
            border-radius: 20px;
            text-decoration: none;
        }
        .no-results a:hover { background: #2980b9; }
        footer {
            background: #2c3e50;
            color: #ecf0f1;
            text-align: center;
            padding: 20px;
            font-size: 14px;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 MyShop</h1>
        <a href="/">Home</a>
    </header>
    <div class="search-bar">
        <form class="search-form" action="/search" method="get">
            <input type="text" name="q" value="{{ query }}" placeholder="Search products...">
            <button type="submit">Search</button>
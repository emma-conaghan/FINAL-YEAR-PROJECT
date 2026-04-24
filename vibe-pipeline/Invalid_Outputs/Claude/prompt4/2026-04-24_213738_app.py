from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)

DB_NAME = "shop.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL
        )
    ''')
    c.execute("SELECT COUNT(*) FROM products")
    count = c.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with great camera", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Powerful Android smartphone", 849.99),
            ("Nike Running Shoes", "Footwear", "Comfortable shoes for running and sports", 89.99),
            ("Adidas Sneakers", "Footwear", "Stylish casual sneakers for everyday wear", 79.99),
            ("Python Programming Book", "Books", "Learn Python from scratch with examples", 39.99),
            ("JavaScript Guide", "Books", "Complete guide to modern JavaScript", 34.99),
            ("Coffee Maker", "Kitchen", "Brews delicious coffee every morning", 49.99),
            ("Blender Pro", "Kitchen", "High speed blender for smoothies and soups", 69.99),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for home workouts", 25.99),
            ("Dumbbells Set", "Sports", "Adjustable dumbbell set for strength training", 119.99),
            ("Laptop Stand", "Electronics", "Ergonomic aluminum laptop stand", 35.99),
            ("Wireless Headphones", "Electronics", "Noise cancelling wireless headphones", 199.99),
            ("Cooking Apron", "Kitchen", "Waterproof apron for cooking and baking", 19.99),
            ("Hiking Boots", "Footwear", "Durable boots for outdoor hiking adventures", 129.99),
            ("Data Science Book", "Books", "Introduction to data science and machine learning", 44.99),
        ]
        c.executemany(
            "INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)",
            sample_products
        )
    conn.commit()
    conn.close()

def search_products(query):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    search_term = f"%{query}%"
    c.execute('''
        SELECT id, name, category, description, price
        FROM products
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        ORDER BY name
    ''', (search_term, search_term, search_term))
    results = c.fetchall()
    conn.close()
    return results

def get_all_products():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, category, description, price FROM products ORDER BY name")
    results = c.fetchall()
    conn.close()
    return results

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
            background-color: #f0f4f8;
            color: #333;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 {
            font-size: 28px;
        }
        header span {
            font-size: 14px;
            opacity: 0.8;
        }
        .hero {
            background: linear-gradient(135deg, #3498db, #2ecc71);
            color: white;
            text-align: center;
            padding: 60px 20px;
        }
        .hero h2 {
            font-size: 36px;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 18px;
            margin-bottom: 30px;
            opacity: 0.9;
        }
        .search-box {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-box input {
            padding: 14px 20px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            width: 400px;
            max-width: 100%;
            outline: none;
        }
        .search-box button {
            padding: 14px 28px;
            font-size: 16px;
            background-color: #e67e22;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .search-box button:hover {
            background-color: #ca6f1e;
        }
        .categories {
            max-width: 1000px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .categories h3 {
            font-size: 22px;
            margin-bottom: 16px;
            color: #2c3e50;
        }
        .category-links {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        .category-links a {
            background-color: #2c3e50;
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 14px;
            transition: background 0.3s;
        }
        .category-links a:hover {
            background-color: #3498db;
        }
        .all-products {
            max-width: 1000px;
            margin: 20px auto 60px;
            padding: 0 20px;
        }
        .all-products h3 {
            font-size: 22px;
            margin-bottom: 16px;
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
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        .product-card h4 {
            font-size: 16px;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .product-card .category-badge {
            background-color: #eaf4fb;
            color: #2980b9;
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 12px;
            display: inline-block;
            margin-bottom: 8px;
        }
        .product-card p {
            font-size: 13px;
            color: #777;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 18px;
            font-weight: bold;
            color: #27ae60;
        }
        footer {
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 MyShop</h1>
        <span>Your favourite online store</span>
    </header>

    <div class="hero">
        <h2>Find What You Need</h2>
        <p>Search by product name, category, or keyword</p>
        <form action="/search" method="get">
            <div class="search-box">
                <input type="text" name="q" placeholder="e.g. shoes, electronics, book..." required>
                <button type="submit">🔍 Search</button>
            </div>
        </form>
    </div>

    <div class="categories">
        <h3>Browse by Category</h3>
        <div class="category-links">
            <a href="/search?q=Electronics">📱 Electronics</a>
            <a href="/search?q=Footwear">👟 Footwear</a>
            <a href="/search?q=Books">📚 Books</a>
            <a href="/search?q=Kitchen">🍳 Kitchen</a>
            <a href="/search?q=Sports">⚽ Sports</a>
        </div>
    </div>

    <div class="all-products">
        <h3>All Products</h3>
        <div class="product-grid">
            {% for product in products %}
            <div class="product-card">
                <h4>{{ product[1] }}</h4>
                <span class="category-badge">{{ product[2] }}</span>
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
'''

RESULTS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results - MyShop</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f4f8;
            color: #333;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 a {
            color: white;
            text-decoration: none;
        }
        .search-bar {
            background-color: #3498db;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }
        .search-bar form {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-bar input {
            padding: 10px 16px;
            font-size: 15px;
            border: none;
            border-radius: 6px;
            width: 300px;
            max-width: 100%;
            outline: none;
        }
        .search-bar button {
            padding: 10px 20px;
            font-size: 15px;
            background-color: #e67e22;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .search-bar button:hover {
            background-color: #ca6f1e;
        }
        .back-link {
            color: white;
            text-decoration: none;
            font-size: 14px;
            opacity: 0.9;
        }
        .back-link:hover {
            opacity: 1;
            text-decoration: underline;
        }
        .results-info {
            max-width: 1000px;
            margin: 30px auto 10px;
            padding: 0 20px;
            font-size: 16px;
            color: #555;
        }
        .results-info strong {
            color: #2c3e50;
        }
        .product-grid {
            max-width: 1000px;
            margin: 20px auto 60px;
            padding: 0 20px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        .product-card h4 {
            font-size: 16px;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .category-badge {
            background-color: #eaf4fb;
            color: #2980b9;
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 12px;
            display: inline-block;
            margin-bottom: 8px;
        }
        .product-card p {
            font-size: 13px;
            color: #777;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .price {
            font-size: 18px;
            font-weight: bold;
            color: #27ae60;
        }
        .no-results {
            max-width: 600px;
            margin: 60px auto;
            text-align: center;
            padding: 0 20px;
        }
        .no-results h2 {
            font-size: 24px;
            color: #e74c3c;
            margin-bottom: 12px;
        }
        .no-results p {
            font-size: 16px;
            color: #777;
            margin-bottom: 20px;
        }
        .no-results a {
            background-color: #3498db;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-size: 15px;
        }
        .no-results a:hover {
            background-color: #2980b9;
        }
        footer {
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <header
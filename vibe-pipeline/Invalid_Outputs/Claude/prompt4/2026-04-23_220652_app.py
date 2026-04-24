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
            ("Samsung Galaxy S23", "Electronics", "Flagship Android phone with great camera", 799.99),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99),
            ("Nike Air Max 270", "Shoes", "Comfortable running shoes with air cushion", 129.99),
            ("Adidas Ultraboost 22", "Shoes", "High performance running shoes", 189.99),
            ("Levi's 501 Jeans", "Clothing", "Classic straight fit denim jeans", 59.99),
            ("Wooden Dining Table", "Furniture", "Solid oak dining table seats 6", 499.99),
            ("Office Chair", "Furniture", "Ergonomic office chair with lumbar support", 249.99),
            ("Python Programming Book", "Books", "Learn Python from scratch beginner guide", 39.99),
            ("Harry Potter Box Set", "Books", "Complete Harry Potter book collection", 89.99),
            ("Coffee Maker Deluxe", "Kitchen", "Programmable drip coffee maker 12 cup", 79.99),
            ("Blender Pro 3000", "Kitchen", "High speed blender for smoothies and soups", 149.99),
            ("Yoga Mat", "Sports", "Non slip eco friendly yoga mat", 34.99),
            ("Dumbbell Set", "Sports", "Adjustable dumbbell set 5 to 50 lbs", 299.99),
            ("Sunglasses Classic", "Accessories", "UV400 protection polarized sunglasses", 49.99),
            ("Leather Wallet", "Accessories", "Slim genuine leather bifold wallet", 29.99),
            ("Gaming Mouse", "Electronics", "High precision wireless gaming mouse", 69.99),
            ("Mechanical Keyboard", "Electronics", "RGB backlit mechanical gaming keyboard", 119.99),
            ("Backpack Travel", "Accessories", "Waterproof laptop backpack 30L capacity", 59.99),
            ("Indoor Plant Pot Set", "Home Decor", "Set of 3 ceramic plant pots with drainage", 24.99),
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
    cursor.execute("SELECT id, name, category, description, price FROM products ORDER BY category, name")
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
            background-color: #f4f4f4;
            color: #333;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 {
            font-size: 24px;
        }
        header a {
            color: white;
            text-decoration: none;
            font-size: 16px;
        }
        .hero {
            background: linear-gradient(135deg, #3498db, #2c3e50);
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
        .search-box input[type="text"] {
            padding: 14px 20px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            width: 400px;
            max-width: 90vw;
            outline: none;
        }
        .search-box button {
            padding: 14px 28px;
            font-size: 16px;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .search-box button:hover {
            background-color: #c0392b;
        }
        .categories-section {
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .categories-section h3 {
            font-size: 22px;
            margin-bottom: 15px;
            color: #2c3e50;
        }
        .category-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .category-btn {
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
            transition: background 0.2s;
        }
        .category-btn:hover {
            background-color: #2980b9;
        }
        .products-section {
            max-width: 1100px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .products-section h3 {
            font-size: 22px;
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
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        .product-card .category-tag {
            display: inline-block;
            background-color: #ecf0f1;
            color: #7f8c8d;
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 12px;
            margin-bottom: 8px;
        }
        .product-card h4 {
            font-size: 16px;
            margin-bottom: 6px;
            color: #2c3e50;
        }
        .product-card p {
            font-size: 13px;
            color: #7f8c8d;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 20px;
            font-weight: bold;
            color: #e74c3c;
        }
        .add-btn {
            display: block;
            margin-top: 12px;
            padding: 8px;
            background-color: #2ecc71;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            width: 100%;
            transition: background 0.2s;
        }
        .add-btn:hover {
            background-color: #27ae60;
        }
        footer {
            text-align: center;
            padding: 20px;
            background-color: #2c3e50;
            color: #bdc3c7;
            margin-top: 50px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ MyShop</h1>
        <a href="/">Home</a>
    </header>

    <div class="hero">
        <h2>Find What You Need</h2>
        <p>Search thousands of products by name, category, or keyword</p>
        <form action="/search" method="get">
            <div class="search-box">
                <input type="text" name="q" placeholder="Search for products, categories, keywords..." autofocus>
                <button type="submit">🔍 Search</button>
            </div>
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
        <h3>All Products</h3>
        <div class="product-grid">
            {% for product in products %}
            <div class="product-card">
                <span class="category-tag">{{ product[2] }}</span>
                <h4>{{ product[1] }}</h4>
                <p>{{ product[3] }}</p>
                <div class="price">${{ "%.2f"|format(product[4]) }}</div>
                <button class="add-btn">Add to Cart</button>
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

SEARCH_TEMPLATE = """
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
            background-color: #f4f4f4;
            color: #333;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 {
            font-size: 24px;
        }
        header a {
            color: white;
            text-decoration: none;
            font-size: 16px;
        }
        .search-bar {
            background-color: #34495e;
            padding: 20px 30px;
        }
        .search-bar form {
            display: flex;
            gap: 10px;
            max-width: 700px;
            margin: 0 auto;
        }
        .search-bar input[type="text"] {
            flex: 1;
            padding: 12px 18px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            outline: none;
        }
        .search-bar button {
            padding: 12px 24px;
            font-size: 16px;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .search-bar button:hover {
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
            font-size: 22px;
            color: #2c3e50;
        }
        .results-header p {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .results-header .query-highlight {
            color: #e74c3c;
            font-weight: bold;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        .product-card .category-tag {
            display: inline-block;
            background-color: #ecf0f1;
            color: #7f8c8d;
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 12px;
            margin-bottom: 8px;
        }
        .product-card h4 {
            font-size: 16px;
            margin-bottom: 6px;
            color: #2c3e50;
        }
        .product-card p {
            font-size: 13px;
            color
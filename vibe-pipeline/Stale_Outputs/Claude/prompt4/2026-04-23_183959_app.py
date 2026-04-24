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
    if c.fetchone()[0] == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with A15 chip", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Flagship Android phone by Samsung", 899.99),
            ("Nike Running Shoes", "Footwear", "Comfortable shoes for running and sports", 129.99),
            ("Adidas Sneakers", "Footwear", "Stylish casual sneakers for everyday wear", 89.99),
            ("Python Programming Book", "Books", "Learn Python from scratch with examples", 39.99),
            ("JavaScript Guide", "Books", "Complete guide to modern JavaScript", 34.99),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker for home use", 59.99),
            ("Blender Pro", "Kitchen", "High speed blender for smoothies and soups", 79.99),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for exercise and meditation", 24.99),
            ("Dumbbells Set", "Sports", "Adjustable dumbbell set for home workouts", 149.99),
            ("Wireless Headphones", "Electronics", "Noise cancelling Bluetooth headphones", 199.99),
            ("USB-C Cable", "Electronics", "Fast charging USB-C cable 2 meter length", 14.99),
            ("Leather Wallet", "Accessories", "Slim genuine leather wallet with card slots", 44.99),
            ("Sunglasses", "Accessories", "UV400 protection stylish sunglasses", 29.99),
            ("Cookware Set", "Kitchen", "Non-stick 10 piece cookware set", 119.99),
            ("Mystery Novel", "Books", "A gripping mystery thriller novel", 12.99),
            ("Basketball", "Sports", "Official size indoor outdoor basketball", 34.99),
            ("Laptop Stand", "Electronics", "Adjustable aluminum laptop stand", 49.99),
            ("Backpack", "Accessories", "Waterproof travel backpack with USB port", 64.99),
            ("Electric Kettle", "Kitchen", "1.7 liter fast boil electric kettle", 35.99),
        ]
        c.executemany("INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)", sample_products)
    conn.commit()
    conn.close()

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Small Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; color: #333; }
        header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        header h1 { font-size: 2em; }
        header p { margin-top: 5px; color: #bdc3c7; }
        .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .search-box { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .search-box h2 { margin-bottom: 20px; color: #2c3e50; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type=text], select {
            width: 100%; padding: 10px 14px; border: 1px solid #ddd;
            border-radius: 5px; font-size: 16px; outline: none;
        }
        input[type=text]:focus, select:focus { border-color: #2980b9; }
        button {
            background: #2980b9; color: white; border: none;
            padding: 12px 30px; border-radius: 5px; font-size: 16px;
            cursor: pointer; width: 100%; margin-top: 5px;
        }
        button:hover { background: #1a6fa8; }
        .categories { margin-top: 40px; }
        .categories h2 { margin-bottom: 15px; color: #2c3e50; }
        .cat-grid { display: flex; flex-wrap: wrap; gap: 10px; }
        .cat-btn {
            background: white; border: 2px solid #2980b9; color: #2980b9;
            padding: 8px 18px; border-radius: 20px; text-decoration: none;
            font-size: 14px; transition: all 0.2s;
        }
        .cat-btn:hover { background: #2980b9; color: white; }
        footer { text-align: center; padding: 30px; color: #999; margin-top: 40px; }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Small Online Shop</h1>
        <p>Find the best products at great prices</p>
    </header>
    <div class="container">
        <div class="search-box">
            <h2>Search Products</h2>
            <form action="/search" method="get">
                <div class="form-group">
                    <label for="query">Search by name or keyword:</label>
                    <input type="text" id="query" name="query" placeholder="e.g. phone, shoes, book..." />
                </div>
                <div class="form-group">
                    <label for="category">Filter by category:</label>
                    <select id="category" name="category">
                        <option value="">-- All Categories --</option>
                        <option value="Electronics">Electronics</option>
                        <option value="Footwear">Footwear</option>
                        <option value="Books">Books</option>
                        <option value="Kitchen">Kitchen</option>
                        <option value="Sports">Sports</option>
                        <option value="Accessories">Accessories</option>
                    </select>
                </div>
                <button type="submit">🔍 Search</button>
            </form>
        </div>

        <div class="categories">
            <h2>Browse by Category</h2>
            <div class="cat-grid">
                <a href="/search?category=Electronics" class="cat-btn">📱 Electronics</a>
                <a href="/search?category=Footwear" class="cat-btn">👟 Footwear</a>
                <a href="/search?category=Books" class="cat-btn">📚 Books</a>
                <a href="/search?category=Kitchen" class="cat-btn">🍳 Kitchen</a>
                <a href="/search?category=Sports" class="cat-btn">⚽ Sports</a>
                <a href="/search?category=Accessories" class="cat-btn">👜 Accessories</a>
            </div>
        </div>
    </div>
    <footer>
        <p>&copy; 2024 Small Online Shop. All rights reserved.</p>
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
    <title>Search Results - Small Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; color: #333; }
        header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        header h1 { font-size: 2em; }
        header p { margin-top: 5px; color: #bdc3c7; }
        .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .search-again { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .search-again form { display: flex; gap: 10px; flex-wrap: wrap; }
        .search-again input[type=text] {
            flex: 1; padding: 10px 14px; border: 1px solid #ddd;
            border-radius: 5px; font-size: 16px; min-width: 200px;
        }
        .search-again select {
            padding: 10px 14px; border: 1px solid #ddd;
            border-radius: 5px; font-size: 16px;
        }
        .search-again button {
            background: #2980b9; color: white; border: none;
            padding: 10px 20px; border-radius: 5px; font-size: 16px; cursor: pointer;
        }
        .search-again button:hover { background: #1a6fa8; }
        .results-info { margin-bottom: 20px; color: #555; font-size: 15px; }
        .results-info span { font-weight: bold; color: #2c3e50; }
        .product-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .product-card {
            background: white; border-radius: 8px; padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.2s;
        }
        .product-card:hover { transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .product-name { font-size: 1.1em; font-weight: bold; color: #2c3e50; margin-bottom: 8px; }
        .product-category {
            display: inline-block; background: #eaf4fb; color: #2980b9;
            padding: 3px 10px; border-radius: 12px; font-size: 12px; margin-bottom: 10px;
        }
        .product-description { color: #666; font-size: 14px; line-height: 1.5; margin-bottom: 15px; }
        .product-price { font-size: 1.4em; font-weight: bold; color: #27ae60; }
        .no-results { text-align: center; padding: 60px 20px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .no-results h2 { color: #7f8c8d; margin-bottom: 10px; }
        .no-results p { color: #999; }
        .back-link { display: inline-block; margin-top: 15px; color: #2980b9; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        footer { text-align: center; padding: 30px; color: #999; margin-top: 40px; }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Small Online Shop</h1>
        <p>Find the best products at great prices</p>
    </header>
    <div class="container">
        <div class="search-again">
            <form action="/search" method="get">
                <input type="text" name="query" placeholder="Search products..." value="{{ query }}" />
                <select name="category">
                    <option value="">All Categories</option>
                    <option value="Electronics" {% if category == 'Electronics' %}selected{% endif %}>Electronics</option>
                    <option value="Footwear" {% if category == 'Footwear' %}selected{% endif %}>Footwear</option>
                    <option value="Books" {% if category == 'Books' %}selected{% endif %}>Books</option>
                    <option value="Kitchen" {% if category == 'Kitchen' %}selected{% endif %}>Kitchen</option>
                    <option value="Sports" {% if category == 'Sports' %}selected{% endif %}>Sports</option>
                    <option value="Accessories" {% if category == 'Accessories' %}selected{% endif %}>Accessories</option>
                </select>
                <button type="submit">🔍 Search</button>
            </form>
        </div>

        {% if results %}
            <p class="results-info">
                Found <span>{{ results|length }} product(s)</span>
                {% if query %} for "<span>{{ query }}</span>"{% endif %}
                {% if category %} in category "<span>{{ category }}</span>"{% endif %}
            </p>
            <div class="product-grid">
                {% for product in results %}
                <div class="product-card">
                    <div class="product-name">{{ product['name'] }}</div>
                    <span class="product-category">{{ product['category'] }}</span>
                    <p class="product-description">{{ product['description'] }}</p>
                    <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="no-results">
                <h2>😕 No products found</h2>
                <p>Try a different search term or browse all categories.</p>
                <a href="/" class="back-link">← Back to Home</a>
            </div>
        {% endif %}
    </div>
    <footer>
        <p>&copy; 2024 Small Online Shop. All rights reserved.</p>
    </footer>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "").strip()

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if query and category:
        sql = """SELECT * FROM products
                 WHERE category = ?
                 AND (name LIKE ? OR description LIKE ?)"""
        like_query = f"%{query}%"
        c.execute(sql, (category, like_query, like_query))
    elif query:
        sql = """SELECT * FROM products
                 WHERE name LIKE ? OR description LIKE ? OR category LIKE ?"""
        like_query = f"%{query}%"
        c.execute(sql, (like_query, like_query, like_query))
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
            ("Samsung Galaxy S23", "Electronics", "Flagship Android smartphone", 899.99),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99),
            ("Nike Air Max 270", "Shoes", "Comfortable running shoes with air cushion", 129.99),
            ("Adidas Ultraboost 22", "Shoes", "High performance running shoes", 149.99),
            ("Levi's 501 Jeans", "Clothing", "Classic straight fit denim jeans", 59.99),
            ("The North Face Jacket", "Clothing", "Waterproof outdoor jacket", 199.99),
            ("Python Programming Book", "Books", "Learn Python from scratch with examples", 39.99),
            ("JavaScript: The Good Parts", "Books", "Essential JavaScript programming guide", 29.99),
            ("Coffee Maker Deluxe", "Kitchen", "12-cup programmable coffee maker", 79.99),
            ("Instant Pot Duo 7-in-1", "Kitchen", "Multi-use pressure cooker and slow cooker", 99.99),
            ("Yoga Mat Premium", "Sports", "Non-slip exercise yoga mat", 34.99),
            ("Dumbbell Set 20kg", "Sports", "Adjustable weight dumbbell set for home gym", 89.99),
            ("Wireless Mouse Logitech", "Electronics", "Ergonomic wireless mouse for computer", 39.99),
            ("Mechanical Keyboard", "Electronics", "RGB backlit mechanical gaming keyboard", 129.99),
            ("Leather Wallet Brown", "Accessories", "Genuine leather bifold wallet", 24.99),
            ("Sunglasses Polarized", "Accessories", "UV400 polarized sunglasses for outdoor", 49.99),
            ("Blender NutriBullet", "Kitchen", "Personal blender for smoothies and shakes", 69.99),
            ("Harry Potter Box Set", "Books", "Complete Harry Potter book series collection", 89.99),
            ("Running Shorts Men", "Clothing", "Lightweight breathable running shorts", 29.99),
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
    <title>Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f4f4; color: #333; }
        header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        header h1 { font-size: 2em; margin-bottom: 5px; }
        header p { font-size: 0.95em; color: #bdc3c7; }
        .search-section {
            max-width: 700px;
            margin: 40px auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .search-section h2 { margin-bottom: 20px; color: #2c3e50; }
        .search-form { display: flex; flex-direction: column; gap: 15px; }
        .form-group { display: flex; flex-direction: column; gap: 5px; }
        label { font-weight: bold; color: #555; }
        input[type="text"], select {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            width: 100%;
        }
        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #3498db;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            font-size: 1em;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn:hover { background: #2980b9; }
        .categories {
            max-width: 700px;
            margin: 0 auto 40px auto;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }
        .category-badge {
            background: #ecf0f1;
            border: 1px solid #bdc3c7;
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            color: #2c3e50;
            font-size: 0.9em;
            transition: background 0.2s;
        }
        .category-badge:hover { background: #3498db; color: white; border-color: #3498db; }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.85em;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <p>Find great products at great prices</p>
    </header>

    <div class="search-section">
        <h2>Search Products</h2>
        <form class="search-form" method="GET" action="/search">
            <div class="form-group">
                <label for="query">Search by name or keyword:</label>
                <input type="text" id="query" name="query" placeholder="e.g. headphones, jacket, book...">
            </div>
            <div class="form-group">
                <label for="category">Filter by category:</label>
                <select id="category" name="category">
                    <option value="">All Categories</option>
                    <option value="Electronics">Electronics</option>
                    <option value="Shoes">Shoes</option>
                    <option value="Clothing">Clothing</option>
                    <option value="Books">Books</option>
                    <option value="Kitchen">Kitchen</option>
                    <option value="Sports">Sports</option>
                    <option value="Accessories">Accessories</option>
                </select>
            </div>
            <button class="btn" type="submit">🔍 Search</button>
        </form>
    </div>

    <div style="text-align:center; margin-bottom:10px; color:#555; font-weight:bold;">Browse by Category</div>
    <div class="categories">
        <a class="category-badge" href="/search?category=Electronics">📱 Electronics</a>
        <a class="category-badge" href="/search?category=Shoes">👟 Shoes</a>
        <a class="category-badge" href="/search?category=Clothing">👕 Clothing</a>
        <a class="category-badge" href="/search?category=Books">📚 Books</a>
        <a class="category-badge" href="/search?category=Kitchen">🍳 Kitchen</a>
        <a class="category-badge" href="/search?category=Sports">🏋️ Sports</a>
        <a class="category-badge" href="/search?category=Accessories">👜 Accessories</a>
    </div>

    <footer>© 2024 My Online Shop. All rights reserved.</footer>
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
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f4f4; color: #333; }
        header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        header h1 { font-size: 2em; }
        header a { color: #3498db; text-decoration: none; font-size: 0.9em; }
        header a:hover { text-decoration: underline; }
        .search-bar-top {
            max-width: 900px;
            margin: 20px auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .search-bar-top form {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: flex-end;
        }
        .search-bar-top input[type="text"] {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            min-width: 150px;
        }
        .search-bar-top select {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1em;
            border-radius: 6px;
            cursor: pointer;
        }
        .btn:hover { background: #2980b9; }
        .btn-secondary {
            background: #95a5a6;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1em;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
        .btn-secondary:hover { background: #7f8c8d; }
        .results-info {
            max-width: 900px;
            margin: 0 auto 15px auto;
            padding: 0 10px;
            color: #555;
            font-size: 0.95em;
        }
        .results-info span { font-weight: bold; color: #2c3e50; }
        .products-grid {
            max-width: 900px;
            margin: 0 auto 40px auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            padding: 0 10px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }
        .product-category {
            font-size: 0.75em;
            background: #eaf4fd;
            color: #2980b9;
            padding: 3px 10px;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .product-name {
            font-size: 1.05em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        .product-description {
            font-size: 0.88em;
            color: #777;
            margin-bottom: 14px;
            line-height: 1.4;
        }
        .product-price {
            font-size: 1.2em;
            color: #27ae60;
            font-weight: bold;
        }
        .no-results {
            max-width: 900px;
            margin: 40px auto;
            text-align: center;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .no-results h2 { color: #e74c3c; margin-bottom: 10px; }
        .no-results p { color: #777; margin-bottom: 20px; }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <p><a href="/">← Back to Home</a></p>
    </header>

    <div class="search-bar-top">
        <form method="GET" action="/search">
            <input type="text" name="query" placeholder="Search products..." value="{{ query }}">
            <select name="category">
                <option value="">All Categories</option>
                {% for cat in categories %}
                <option value="{{ cat }}" {% if cat == selected_category %}selected{% endif %}>{{ cat }}</option>
                {% endfor %}
            </select>
            <button class="btn" type="submit">🔍 Search</button>
            <a class="btn-secondary" href="/">Clear</a>
        </form>
    </div>

    {% if products %}
    <div class="results-info">
        Found <span>{{ products|length }}</span> result(s)
        {% if query %} for "<span>{{ query }}</span>"{% endif %}
        {% if selected_category %} in category "<span>{{ selected_category }}</span>"{% endif %}
    </div>
    <div class="products-grid">
        {% for product in products %}
        <div class="product-card">
            <div class="product-category">{{ product['category'] }}</div>
            <div class="product-name">{{ product['name'] }}</div>
            <div class="product-description">{{ product['description'] }}</div>
            <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="no-results">
        <h2>😕 No Products Found
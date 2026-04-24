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
            ("Apple iPhone 15", "Electronics", "Latest Apple smartphone with A17 chip", 999.99),
            ("Samsung Galaxy S24", "Electronics", "Flagship Android phone with AI features", 849.99),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99),
            ("Nike Air Max 270", "Shoes", "Comfortable running shoes with air cushioning", 129.99),
            ("Adidas Ultraboost 22", "Shoes", "High performance running shoes", 189.99),
            ("Levi's 501 Jeans", "Clothing", "Classic straight fit denim jeans", 59.99),
            ("Cotton T-Shirt", "Clothing", "Soft and comfortable everyday t-shirt", 19.99),
            ("Python Programming Book", "Books", "Learn Python programming from scratch", 39.99),
            ("JavaScript for Beginners", "Books", "Complete guide to modern JavaScript", 34.99),
            ("Coffee Maker Deluxe", "Kitchen", "12-cup programmable coffee maker", 79.99),
            ("Blender Pro 5000", "Kitchen", "High speed blender for smoothies and more", 99.99),
            ("Yoga Mat Premium", "Sports", "Non-slip thick yoga mat for exercise", 29.99),
            ("Dumbbells Set 20kg", "Sports", "Adjustable dumbbell set for home gym", 149.99),
            ("Sunglasses UV400", "Accessories", "Stylish sunglasses with UV protection", 49.99),
            ("Leather Wallet", "Accessories", "Genuine leather bifold wallet", 39.99),
        ]
        c.executemany("INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)", sample_products)
    conn.commit()
    conn.close()

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f4f4; color: #333; }
        header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        header h1 { font-size: 2em; }
        header p { margin-top: 5px; color: #bdc3c7; }
        .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .search-box { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .search-box h2 { margin-bottom: 20px; color: #2c3e50; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="text"], select {
            width: 100%; padding: 10px 14px; border: 1px solid #ddd;
            border-radius: 5px; font-size: 1em;
        }
        input[type="text"]:focus, select:focus { outline: none; border-color: #3498db; }
        button {
            background: #3498db; color: white; border: none;
            padding: 12px 30px; font-size: 1em; border-radius: 5px;
            cursor: pointer; margin-top: 10px;
        }
        button:hover { background: #2980b9; }
        .categories { margin-top: 40px; }
        .categories h2 { margin-bottom: 15px; color: #2c3e50; }
        .category-grid { display: flex; flex-wrap: wrap; gap: 10px; }
        .category-btn {
            background: white; border: 2px solid #3498db; color: #3498db;
            padding: 8px 18px; border-radius: 20px; cursor: pointer;
            text-decoration: none; font-size: 0.9em;
        }
        .category-btn:hover { background: #3498db; color: white; }
        footer { text-align: center; padding: 20px; color: #999; margin-top: 40px; font-size: 0.9em; }
    </style>
</head>
<body>
<header>
    <h1>🛒 My Online Shop</h1>
    <p>Find great products at great prices</p>
</header>
<div class="container">
    <div class="search-box">
        <h2>Search Products</h2>
        <form action="/search" method="get">
            <div class="form-group">
                <label for="query">Search by name or keyword:</label>
                <input type="text" id="query" name="query" placeholder="e.g. headphones, book, shoes...">
            </div>
            <div class="form-group">
                <label for="category">Filter by category (optional):</label>
                <select id="category" name="category">
                    <option value="">-- All Categories --</option>
                    <option value="Electronics">Electronics</option>
                    <option value="Shoes">Shoes</option>
                    <option value="Clothing">Clothing</option>
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
        <div class="category-grid">
            <a href="/search?category=Electronics" class="category-btn">📱 Electronics</a>
            <a href="/search?category=Shoes" class="category-btn">👟 Shoes</a>
            <a href="/search?category=Clothing" class="category-btn">👕 Clothing</a>
            <a href="/search?category=Books" class="category-btn">📚 Books</a>
            <a href="/search?category=Kitchen" class="category-btn">🍳 Kitchen</a>
            <a href="/search?category=Sports" class="category-btn">🏋️ Sports</a>
            <a href="/search?category=Accessories" class="category-btn">👓 Accessories</a>
        </div>
    </div>
</div>
<footer>
    <p>© 2024 My Online Shop. All rights reserved.</p>
</footer>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results - My Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f4f4; color: #333; }
        header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        header h1 { font-size: 2em; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .search-again { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; }
        .search-again form { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; }
        .search-again .form-group { flex: 1; min-width: 150px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; font-size: 0.9em; }
        input[type="text"], select {
            width: 100%; padding: 9px 12px; border: 1px solid #ddd;
            border-radius: 5px; font-size: 1em;
        }
        button {
            background: #3498db; color: white; border: none;
            padding: 10px 20px; font-size: 1em; border-radius: 5px;
            cursor: pointer;
        }
        button:hover { background: #2980b9; }
        .results-header { margin-bottom: 15px; }
        .results-header h2 { color: #2c3e50; }
        .results-header p { color: #777; margin-top: 5px; }
        .product-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; }
        .product-card {
            background: white; border-radius: 8px; padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.2s;
        }
        .product-card:hover { transform: translateY(-3px); }
        .product-category {
            display: inline-block; background: #ecf0f1; color: #7f8c8d;
            font-size: 0.75em; padding: 3px 8px; border-radius: 10px; margin-bottom: 8px;
        }
        .product-name { font-size: 1.1em; font-weight: bold; color: #2c3e50; margin-bottom: 8px; }
        .product-desc { font-size: 0.9em; color: #777; margin-bottom: 12px; line-height: 1.4; }
        .product-price { font-size: 1.3em; font-weight: bold; color: #27ae60; }
        .add-btn {
            display: block; text-align: center; background: #e8f5e9;
            color: #27ae60; border: 1px solid #27ae60;
            padding: 8px; border-radius: 5px; margin-top: 12px;
            cursor: pointer; font-size: 0.9em;
        }
        .add-btn:hover { background: #27ae60; color: white; }
        .no-results {
            text-align: center; padding: 60px 20px; background: white;
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .no-results p { font-size: 1.1em; color: #777; margin-bottom: 15px; }
        .back-link { color: #3498db; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        footer { text-align: center; padding: 20px; color: #999; margin-top: 40px; font-size: 0.9em; }
    </style>
</head>
<body>
<header>
    <h1>🛒 My Online Shop</h1>
</header>
<div class="container">
    <div class="search-again">
        <form action="/search" method="get">
            <div class="form-group">
                <label for="query">Search:</label>
                <input type="text" id="query" name="query" value="{{ query }}" placeholder="Search products...">
            </div>
            <div class="form-group">
                <label for="category">Category:</label>
                <select id="category" name="category">
                    <option value="">-- All --</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}" {% if cat == selected_category %}selected{% endif %}>{{ cat }}</option>
                    {% endfor %}
                </select>
            </div>
            <button type="submit">🔍 Search</button>
        </form>
    </div>

    <div class="results-header">
        <h2>Search Results</h2>
        {% if query or selected_category %}
        <p>
            {% if query %}Searching for "<strong>{{ query }}</strong>"{% endif %}
            {% if selected_category %} in <strong>{{ selected_category }}</strong>{% endif %}
            — {{ products|length }} result(s) found
        </p>
        {% else %}
        <p>Showing all {{ products|length }} products</p>
        {% endif %}
    </div>

    {% if products %}
    <div class="product-grid">
        {% for product in products %}
        <div class="product-card">
            <span class="product-category">{{ product['category'] }}</span>
            <div class="product-name">{{ product['name'] }}</div>
            <div class="product-desc">{{ product['description'] }}</div>
            <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
            <div class="add-btn">🛒 Add to Cart</div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="no-results">
        <p>😕 No products found matching your search.</p>
        <a href="/" class="back-link">← Go back and try a different search</a>
    </div>
    {% endif %}

    <div style="margin-top: 25px;">
        <a href="/" class="back-link">← Back to Home</a>
    </div>
</div>
<footer>
    <p>© 2024 My Online Shop. All rights reserved.</p>
</footer>
</body>
</html>
"""

def search_products(query, category):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    conditions = []
    params = []

    if query:
        conditions.append("(name LIKE ? OR description LIKE ? OR category LIKE ?)")
        like_query = f"%{query}%"
        params.extend([like_query, like_query, like_query])

    if category:
        conditions.append("category = ?")
        params.append(category)

    sql = "SELECT * FROM products"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY name"

    c.execute(sql, params)
    results = c.fetchall()
    conn.close()
    return results

@app.route("/")
def home():
    return render_template_string(HOME_HTML)

@app.route("/search")
def search():
    query =
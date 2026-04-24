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
            ("Samsung Galaxy S23", "Electronics", "Android smartphone with amazing display", 849.99),
            ("Nike Running Shoes", "Footwear", "Comfortable shoes for running and jogging", 120.00),
            ("Adidas Sneakers", "Footwear", "Stylish sneakers for everyday wear", 95.00),
            ("Python Programming Book", "Books", "Learn Python programming from scratch", 39.99),
            ("JavaScript Guide", "Books", "Complete guide to modern JavaScript", 34.99),
            ("Coffee Maker", "Kitchen", "Brews delicious coffee every morning", 59.99),
            ("Blender Pro", "Kitchen", "Powerful blender for smoothies and soups", 79.99),
            ("Gaming Headset", "Electronics", "Immersive sound for gaming sessions", 149.99),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for all fitness levels", 25.00),
            ("Dumbbell Set", "Sports", "Adjustable dumbbell set for home workouts", 199.99),
            ("Laptop Stand", "Office", "Ergonomic stand to raise your laptop screen", 45.00),
            ("Mechanical Keyboard", "Office", "Tactile keyboard for fast and accurate typing", 110.00),
            ("Wireless Mouse", "Office", "Smooth and precise wireless mouse", 35.00),
            ("Water Bottle", "Sports", "Insulated bottle keeps drinks cold for 24 hours", 22.00),
            ("Sunglasses", "Accessories", "UV protection sunglasses for outdoor activities", 55.00),
            ("Backpack", "Accessories", "Durable backpack with multiple compartments", 65.00),
            ("Cooking Pan Set", "Kitchen", "Non-stick pan set for everyday cooking", 85.00),
            ("Fiction Novel", "Books", "A thrilling adventure story you cannot put down", 15.99),
            ("Smartwatch", "Electronics", "Track fitness and notifications on your wrist", 250.00),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)",
            sample_products
        )
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
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 40px auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h2 {
            color: #2c3e50;
        }
        input[type="text"] {
            width: 70%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        select {
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-left: 5px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #2c3e50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 5px;
        }
        button:hover {
            background-color: #1a252f;
        }
        .categories {
            margin-top: 30px;
        }
        .categories a {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        .categories a:hover {
            background-color: #2980b9;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Small Online Shop</h1>
        <p>Find the best products at great prices</p>
    </header>
    <div class="container">
        <h2>Search for Products</h2>
        <form method="GET" action="/search">
            <input type="text" name="query" placeholder="Search by name, category, or keyword..." value="{{ query }}">
            <select name="category">
                <option value="">All Categories</option>
                {% for cat in categories %}
                <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                {% endfor %}
            </select>
            <button type="submit">Search</button>
        </form>

        <div class="categories">
            <h3>Browse by Category</h3>
            {% for cat in categories %}
            <a href="/search?category={{ cat }}">{{ cat }}</a>
            {% endfor %}
        </div>
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
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .container {
            max-width: 900px;
            margin: 30px auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h2 {
            color: #2c3e50;
        }
        .search-bar {
            margin-bottom: 20px;
        }
        input[type="text"] {
            width: 60%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        select {
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-left: 5px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #2c3e50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 5px;
        }
        button:hover {
            background-color: #1a252f;
        }
        .back-link {
            display: inline-block;
            margin-bottom: 15px;
            color: #3498db;
            text-decoration: none;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        .result-count {
            color: #666;
            margin-bottom: 15px;
        }
        .product-card {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .product-info h3 {
            margin: 0 0 5px 0;
            color: #2c3e50;
        }
        .product-info p {
            margin: 3px 0;
            color: #555;
            font-size: 14px;
        }
        .product-category {
            display: inline-block;
            background-color: #ecf0f1;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        .product-price {
            font-size: 22px;
            font-weight: bold;
            color: #e74c3c;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        .no-results h3 {
            font-size: 22px;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Small Online Shop</h1>
        <p>Find the best products at great prices</p>
    </header>
    <div class="container">
        <a class="back-link" href="/">← Back to Home</a>

        <div class="search-bar">
            <form method="GET" action="/search">
                <input type="text" name="query" placeholder="Search products..." value="{{ query }}">
                <select name="category">
                    <option value="">All Categories</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                    {% endfor %}
                </select>
                <button type="submit">Search</button>
            </form>
        </div>

        {% if query or selected_category %}
        <h2>Search Results</h2>
        <p class="result-count">
            Found <strong>{{ products|length }}</strong> result(s)
            {% if query %} for "<em>{{ query }}</em>"{% endif %}
            {% if selected_category %} in category "<em>{{ selected_category }}</em>"{% endif %}
        </p>
        {% else %}
        <h2>All Products</h2>
        <p class="result-count">Showing all <strong>{{ products|length }}</strong> products</p>
        {% endif %}

        {% if products %}
            {% for product in products %}
            <div class="product-card">
                <div class="product-info">
                    <h3>{{ product['name'] }}</h3>
                    <p>{{ product['description'] }}</p>
                    <span class="product-category">{{ product['category'] }}</span>
                </div>
                <div class="product-price">${{ "%.2f" | format(product['price']) }}</div>
            </div>
            {% endfor %}
        {% else %}
            <div class="no-results">
                <h3>😕 No products found</h3>
                <p>Try searching with different keywords or browse all categories.</p>
                <a href="/search">View All Products</a>
            </div>
        {% endif %}
    </div>
    <footer>
        <p>&copy; 2024 Small Online Shop. All rights reserved.</p>
    </footer>
</body>
</html>
"""

def get_categories():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM products ORDER BY category")
    rows = cursor.fetchall()
    conn.close()
    return [row["category"] for row in rows]

def search_products(query="", category=""):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sql = "SELECT * FROM products WHERE 1=1"
    params = []

    if query:
        sql += " AND (name LIKE ? OR category LIKE ? OR description LIKE ?)"
        like_query = f"%{query}%"
        params.extend([like_query, like_query, like_query])

    if category:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY name"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.route("/")
def home():
    categories = get_categories()
    query = request.args.get("query", "")
    selected_category = request.args.get("category", "")
    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        query=query,
        selected_category=selected_category
    )

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    selected_category = request.args.get("category", "").strip()
    categories = get_categories()
    products = search_products(query=query, category=selected_category)
    return render_template_string(
        SEARCH_TEMPLATE,
        products=products,
        query=query,
        selected_category=selected_category,
        categories=categories
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
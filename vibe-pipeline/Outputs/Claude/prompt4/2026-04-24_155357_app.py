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
            price REAL NOT NULL,
            keyword TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with A15 chip", 999.99, "phone mobile apple"),
            ("Samsung Galaxy S23", "Electronics", "Android smartphone with great camera", 849.99, "phone mobile samsung android"),
            ("Nike Running Shoes", "Footwear", "Lightweight running shoes for all terrains", 89.99, "shoes sport running nike"),
            ("Adidas Sneakers", "Footwear", "Casual sneakers for everyday wear", 69.99, "shoes casual adidas sneakers"),
            ("Python Programming Book", "Books", "Learn Python from scratch with this beginner guide", 29.99, "python programming coding book"),
            ("JavaScript Guide", "Books", "Complete guide to modern JavaScript development", 34.99, "javascript web coding book"),
            ("Coffee Maker", "Kitchen", "Brews perfect coffee every morning", 49.99, "coffee kitchen appliance brew"),
            ("Blender Pro", "Kitchen", "High speed blender for smoothies and soups", 59.99, "blender kitchen smoothie"),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for home workouts", 24.99, "yoga mat exercise fitness"),
            ("Dumbbells Set", "Sports", "Adjustable dumbbells for strength training", 79.99, "weights gym fitness training"),
            ("Laptop Stand", "Office", "Ergonomic laptop stand for better posture", 39.99, "laptop stand desk office ergonomic"),
            ("Wireless Mouse", "Office", "Compact wireless mouse with long battery life", 25.99, "mouse wireless computer office"),
            ("Headphones", "Electronics", "Noise cancelling over-ear headphones", 149.99, "headphones audio music noise"),
            ("Backpack", "Accessories", "Durable backpack for travel and daily use", 44.99, "bag backpack travel accessories"),
            ("Sunglasses", "Accessories", "UV protected sunglasses for outdoor activities", 19.99, "sunglasses outdoor accessories uv"),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, description, price, keyword) VALUES (?, ?, ?, ?, ?)",
            sample_products
        )
    conn.commit()
    conn.close()

def search_products(query):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    search_term = f"%{query}%"
    cursor.execute("""
        SELECT id, name, category, description, price FROM products
        WHERE name LIKE ?
           OR category LIKE ?
           OR keyword LIKE ?
           OR description LIKE ?
    """, (search_term, search_term, search_term, search_term))
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
    cats = [row[0] for row in cursor.fetchall()]
    conn.close()
    return cats

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f6f9; color: #333; }
        header {
            background: #2c3e50;
            color: white;
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 1.8em; }
        header a { color: #ecf0f1; text-decoration: none; font-size: 0.95em; }
        .search-section {
            background: #3498db;
            padding: 50px 20px;
            text-align: center;
            color: white;
        }
        .search-section h2 { font-size: 2em; margin-bottom: 10px; }
        .search-section p { margin-bottom: 25px; font-size: 1.1em; opacity: 0.9; }
        .search-form {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-form input[type=text] {
            padding: 12px 20px;
            font-size: 1em;
            border: none;
            border-radius: 25px;
            width: 350px;
            outline: none;
        }
        .search-form button {
            padding: 12px 30px;
            font-size: 1em;
            background: #2c3e50;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
        }
        .search-form button:hover { background: #1a252f; }
        .categories {
            padding: 30px 40px;
            background: white;
            border-bottom: 1px solid #ddd;
        }
        .categories h3 { margin-bottom: 15px; color: #555; }
        .category-list { display: flex; gap: 10px; flex-wrap: wrap; }
        .category-btn {
            padding: 8px 18px;
            background: #ecf0f1;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            text-decoration: none;
            color: #333;
            transition: background 0.2s;
        }
        .category-btn:hover { background: #3498db; color: white; }
        .products-section { padding: 30px 40px; }
        .products-section h3 { margin-bottom: 20px; color: #555; font-size: 1.2em; }
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
        .product-card:hover { transform: translateY(-4px); box-shadow: 0 6px 16px rgba(0,0,0,0.12); }
        .product-card .category-tag {
            font-size: 0.75em;
            background: #eaf4fb;
            color: #3498db;
            padding: 3px 10px;
            border-radius: 12px;
            display: inline-block;
            margin-bottom: 8px;
        }
        .product-card h4 { font-size: 1em; margin-bottom: 6px; }
        .product-card p { font-size: 0.85em; color: #777; margin-bottom: 10px; }
        .product-card .price { font-size: 1.2em; font-weight: bold; color: #27ae60; }
        footer {
            background: #2c3e50;
            color: #aaa;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            font-size: 0.9em;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .no-results h3 { font-size: 1.5em; margin-bottom: 10px; }
    </style>
</head>
<body>

<header>
    <h1>🛒 MyShop</h1>
    <a href="/">Home</a>
</header>

<div class="search-section">
    <h2>Find What You Need</h2>
    <p>Search by product name, category, or keyword</p>
    <form class="search-form" method="GET" action="/search">
        <input type="text" name="q" placeholder="e.g. phone, shoes, coffee..." value="{{ query or '' }}" autofocus>
        <button type="submit">🔍 Search</button>
    </form>
</div>

<div class="categories">
    <h3>Browse by Category:</h3>
    <div class="category-list">
        {% for cat in categories %}
        <a href="/search?q={{ cat }}" class="category-btn">{{ cat }}</a>
        {% endfor %}
    </div>
</div>

<div class="products-section">
    {% if query %}
        <h3>Search results for "<strong>{{ query }}</strong>" — {{ products|length }} item(s) found</h3>
    {% else %}
        <h3>All Products ({{ products|length }} items)</h3>
    {% endif %}

    {% if products %}
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
    {% else %}
    <div class="no-results">
        <h3>😕 No products found</h3>
        <p>Try a different search term or browse by category above.</p>
    </div>
    {% endif %}
</div>

<footer>
    <p>&copy; 2024 MyShop — All rights reserved</p>
</footer>

</body>
</html>
"""

@app.route("/")
def home():
    products = get_all_products()
    categories = get_categories()
    return render_template_string(HOME_TEMPLATE, products=products, categories=categories, query=None)

@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    categories = get_categories()
    if query:
        products = search_products(query)
    else:
        products = get_all_products()
    return render_template_string(HOME_TEMPLATE, products=products, categories=categories, query=query if query else None)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
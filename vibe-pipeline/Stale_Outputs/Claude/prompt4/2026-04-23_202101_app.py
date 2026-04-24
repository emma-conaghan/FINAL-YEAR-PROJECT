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
            ("Samsung Galaxy S23", "Electronics", "Android flagship phone with powerful processor", 849.99),
            ("Nike Air Max", "Shoes", "Comfortable running shoes with air cushion", 129.99),
            ("Adidas Ultraboost", "Shoes", "High performance running shoes", 179.99),
            ("Levi's 501 Jeans", "Clothing", "Classic straight fit denim jeans", 59.99),
            ("The North Face Jacket", "Clothing", "Waterproof outdoor jacket for hiking", 199.99),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99),
            ("Kindle Paperwhite", "Electronics", "E-reader with adjustable warm light", 139.99),
            ("Yoga Mat", "Sports", "Non-slip exercise mat for yoga and fitness", 29.99),
            ("Dumbbells Set", "Sports", "Adjustable weight dumbbells for home gym", 89.99),
            ("Coffee Maker", "Kitchen", "Programmable drip coffee maker", 49.99),
            ("Instant Pot", "Kitchen", "Multi-use pressure cooker and slow cooker", 79.99),
            ("Harry Potter Book Set", "Books", "Complete 7-book series collection", 89.99),
            ("Python Programming Book", "Books", "Learn Python from scratch beginner guide", 39.99),
            ("Backpack", "Accessories", "Durable travel backpack with laptop compartment", 69.99),
            ("Sunglasses", "Accessories", "UV400 protection polarized sunglasses", 24.99),
            ("Bluetooth Speaker", "Electronics", "Portable waterproof outdoor speaker", 59.99),
            ("Running Shorts", "Clothing", "Lightweight breathable athletic shorts", 34.99),
            ("Tennis Racket", "Sports", "Beginner to intermediate tennis racket", 49.99),
            ("Blender", "Kitchen", "High speed smoothie and food blender", 39.99),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)",
            sample_products
        )
    conn.commit()
    conn.close()

def search_products(query, category):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    results = []
    if query and category:
        cursor.execute("""
            SELECT id, name, category, description, price FROM products
            WHERE (name LIKE ? OR description LIKE ?) AND category = ?
        """, (f"%{query}%", f"%{query}%", category))
    elif query:
        cursor.execute("""
            SELECT id, name, category, description, price FROM products
            WHERE name LIKE ? OR description LIKE ? OR category LIKE ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%"))
    elif category:
        cursor.execute("""
            SELECT id, name, category, description, price FROM products
            WHERE category = ?
        """, (category,))
    else:
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
            padding: 20px 40px;
            text-align: center;
        }
        header h1 {
            font-size: 2em;
            margin-bottom: 5px;
        }
        header p {
            font-size: 1em;
            opacity: 0.8;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .search-box {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .search-box h2 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .search-form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        label {
            font-weight: bold;
            color: #555;
        }
        input[type="text"], select {
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            width: 100%;
        }
        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #2c3e50;
        }
        button {
            background-color: #e74c3c;
            color: white;
            border: none;
            padding: 14px 30px;
            font-size: 1em;
            border-radius: 6px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #c0392b;
        }
        .results-info {
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.1);
            color: #555;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
        }
        .product-name {
            font-size: 1.1em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        .product-category {
            display: inline-block;
            background-color: #ecf0f1;
            color: #7f8c8d;
            font-size: 0.8em;
            padding: 3px 8px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .product-description {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 15px;
            line-height: 1.4;
        }
        .product-price {
            font-size: 1.3em;
            font-weight: bold;
            color: #e74c3c;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 10px;
            color: #999;
        }
        .no-results h3 {
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #aaa;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ Small Online Shop</h1>
        <p>Find great products at great prices</p>
    </header>
    <div class="container">
        <div class="search-box">
            <h2>Search Products</h2>
            <form class="search-form" method="GET" action="/search">
                <div class="form-group">
                    <label for="query">Search by name or keyword:</label>
                    <input type="text" id="query" name="query" placeholder="e.g. headphones, running, coffee..." value="{{ query }}">
                </div>
                <div class="form-group">
                    <label for="category">Filter by category:</label>
                    <select id="category" name="category">
                        <option value="">All Categories</option>
                        {% for cat in categories %}
                        <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                        {% endfor %}
                    </select>
                </div>
                <button type="submit">🔍 Search Products</button>
            </form>
        </div>

        {% if searched %}
        <div class="results-info">
            {% if results %}
            Found <strong>{{ results|length }}</strong> product(s)
            {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
            {% if selected_category %} in category "<strong>{{ selected_category }}</strong>"{% endif %}
            {% else %}
            No products found
            {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
            {% if selected_category %} in category "<strong>{{ selected_category }}</strong>"{% endif %}
            {% endif %}
        </div>

        {% if results %}
        <div class="product-grid">
            {% for product in results %}
            <div class="product-card">
                <div class="product-name">{{ product[1] }}</div>
                <span class="product-category">{{ product[2] }}</span>
                <div class="product-description">{{ product[3] }}</div>
                <div class="product-price">${{ "%.2f"|format(product[4]) }}</div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <h3>😕 No results found</h3>
            <p>Try a different keyword or browse all categories.</p>
        </div>
        {% endif %}
        {% else %}
        <div class="results-info">
            Browse all our products below or use the search box above.
        </div>
        <div class="product-grid">
            {% for product in all_products %}
            <div class="product-card">
                <div class="product-name">{{ product[1] }}</div>
                <span class="product-category">{{ product[2] }}</span>
                <div class="product-description">{{ product[3] }}</div>
                <div class="product-price">${{ "%.2f"|format(product[4]) }}</div>
            </div>
            {% endfor %}
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
    categories = get_categories()
    all_products = search_products("", "")
    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        all_products=all_products,
        results=[],
        query="",
        selected_category="",
        searched=False
    )

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "").strip()
    categories = get_categories()
    results = search_products(query, category)
    all_products = search_products("", "")
    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        all_products=all_products,
        results=results,
        query=query,
        selected_category=category,
        searched=True
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
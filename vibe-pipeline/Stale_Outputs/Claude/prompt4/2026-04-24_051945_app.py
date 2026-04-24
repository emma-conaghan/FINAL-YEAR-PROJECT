from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)

DB_NAME = "shop.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL
        )
    """)
    c.execute("SELECT COUNT(*) FROM products")
    count = c.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with great camera", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Powerful Android smartphone", 849.99),
            ("Running Shoes", "Footwear", "Comfortable shoes for running and jogging", 59.99),
            ("Leather Wallet", "Accessories", "Slim genuine leather wallet", 29.99),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker for home", 49.99),
            ("Yoga Mat", "Sports", "Non-slip exercise yoga mat", 24.99),
            ("Bluetooth Headphones", "Electronics", "Wireless noise-cancelling headphones", 199.99),
            ("Backpack", "Bags", "Durable travel backpack with multiple compartments", 39.99),
            ("Sunglasses", "Accessories", "UV protection stylish sunglasses", 19.99),
            ("Electric Kettle", "Kitchen", "Fast boiling electric kettle 1.7L", 34.99),
            ("Sneakers", "Footwear", "Casual everyday sneakers", 44.99),
            ("Gaming Mouse", "Electronics", "High precision gaming mouse with RGB lighting", 39.99),
            ("Wooden Cutting Board", "Kitchen", "Large wooden cutting board for cooking", 22.99),
            ("Gym Gloves", "Sports", "Protective gloves for weightlifting", 14.99),
            ("Laptop Bag", "Bags", "Professional laptop bag fits up to 15 inch", 54.99),
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
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        header {
            background-color: #333;
            color: white;
            padding: 15px 30px;
            text-align: center;
        }
        header h1 {
            margin: 0;
            font-size: 28px;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h2 {
            color: #333;
            text-align: center;
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-top: 20px;
        }
        input[type="text"], select {
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            padding: 12px;
            font-size: 16px;
            background-color: #333;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #555;
        }
        .categories {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
            justify-content: center;
        }
        .category-tag {
            background-color: #e0e0e0;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            text-decoration: none;
            color: #333;
        }
        .category-tag:hover {
            background-color: #333;
            color: white;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ Small Online Shop</h1>
    </header>
    <div class="container">
        <h2>Search for Products</h2>
        <form action="/search" method="get">
            <input type="text" name="query" placeholder="Search by name or keyword..." value="">
            <select name="category">
                <option value="">All Categories</option>
                {% for cat in categories %}
                <option value="{{ cat }}">{{ cat }}</option>
                {% endfor %}
            </select>
            <button type="submit">Search</button>
        </form>
        <div class="categories">
            <p style="width:100%; text-align:center; color:#666;">Browse by category:</p>
            {% for cat in categories %}
            <a class="category-tag" href="/search?category={{ cat }}&query=">{{ cat }}</a>
            {% endfor %}
        </div>
    </div>
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
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        header {
            background-color: #333;
            color: white;
            padding: 15px 30px;
            text-align: center;
        }
        header h1 {
            margin: 0;
            font-size: 28px;
        }
        .container {
            max-width: 900px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .search-bar {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .search-bar form {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-bar input[type="text"] {
            flex: 1;
            padding: 10px;
            font-size: 15px;
            border: 1px solid #ccc;
            border-radius: 5px;
            min-width: 200px;
        }
        .search-bar select {
            padding: 10px;
            font-size: 15px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .search-bar button {
            padding: 10px 20px;
            font-size: 15px;
            background-color: #333;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .search-bar button:hover {
            background-color: #555;
        }
        .results-info {
            color: #666;
            margin-bottom: 15px;
            font-size: 15px;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
        }
        .product-name {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        .product-category {
            display: inline-block;
            background-color: #e0e0e0;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            color: #555;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 14px;
            margin-bottom: 12px;
            line-height: 1.5;
        }
        .product-price {
            font-size: 20px;
            font-weight: bold;
            color: #27ae60;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 18px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .back-link {
            display: inline-block;
            margin-top: 20px;
            color: #333;
            text-decoration: none;
            font-size: 15px;
        }
        .back-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ Small Online Shop</h1>
    </header>
    <div class="container">
        <div class="search-bar">
            <form action="/search" method="get">
                <input type="text" name="query" placeholder="Search by name or keyword..." value="{{ query }}">
                <select name="category">
                    <option value="">All Categories</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}" {% if cat == selected_category %}selected{% endif %}>{{ cat }}</option>
                    {% endfor %}
                </select>
                <button type="submit">Search</button>
            </form>
        </div>

        <div class="results-info">
            {% if query or selected_category %}
                Found <strong>{{ products|length }}</strong> result(s)
                {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
                {% if selected_category %} in category "<strong>{{ selected_category }}</strong>"{% endif %}
            {% else %}
                Showing all <strong>{{ products|length }}</strong> products
            {% endif %}
        </div>

        {% if products %}
        <div class="product-grid">
            {% for product in products %}
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
            😕 No products found. Try a different search.
        </div>
        {% endif %}

        <a class="back-link" href="/">← Back to Home</a>
    </div>
</body>
</html>
"""

def get_categories():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT DISTINCT category FROM products ORDER BY category")
    rows = c.fetchall()
    conn.close()
    return [row["category"] for row in rows]

def search_products(query, category):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    sql = "SELECT * FROM products WHERE 1=1"
    params = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like_query = "%" + query + "%"
        params.extend([like_query, like_query, like_query])

    if category:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY name"

    c.execute(sql, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.route("/")
def home():
    categories = get_categories()
    return render_template_string(HOME_TEMPLATE, categories=categories)

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "").strip()
    categories = get_categories()
    products = search_products(query, category)
    return render_template_string(
        RESULTS_TEMPLATE,
        products=products,
        query=query,
        selected_category=category,
        categories=categories
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
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
            ("Samsung Galaxy S23", "Electronics", "Android smartphone with great camera", 849.99),
            ("Nike Air Max", "Shoes", "Comfortable running shoes", 129.99),
            ("Adidas Ultraboost", "Shoes", "High performance running shoes", 179.99),
            ("Python Programming Book", "Books", "Learn Python from scratch", 39.99),
            ("JavaScript Guide", "Books", "Complete JavaScript reference guide", 44.99),
            ("Coffee Maker", "Kitchen", "Brew perfect coffee every morning", 59.99),
            ("Blender Pro", "Kitchen", "High speed blender for smoothies", 89.99),
            ("Gaming Mouse", "Electronics", "Precision gaming mouse with RGB", 49.99),
            ("Mechanical Keyboard", "Electronics", "Tactile mechanical keyboard for typing", 119.99),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for home workouts", 29.99),
            ("Dumbbell Set", "Sports", "Adjustable dumbbell set for strength training", 199.99),
            ("Desk Lamp", "Home", "LED desk lamp with adjustable brightness", 34.99),
            ("Backpack", "Accessories", "Waterproof backpack with laptop compartment", 79.99),
            ("Sunglasses", "Accessories", "UV protection polarized sunglasses", 59.99),
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
    <title>Online Shop</title>
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
        header h1 {
            margin: 0;
            font-size: 2em;
        }
        .container {
            max-width: 800px;
            margin: 40px auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .search-box input[type="text"] {
            flex: 1;
            padding: 12px;
            font-size: 1em;
            border: 2px solid #ccc;
            border-radius: 4px;
        }
        .search-box select {
            padding: 12px;
            font-size: 1em;
            border: 2px solid #ccc;
            border-radius: 4px;
        }
        .search-box button {
            padding: 12px 24px;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 1em;
            cursor: pointer;
        }
        .search-box button:hover {
            background-color: #c0392b;
        }
        .categories {
            margin-top: 20px;
        }
        .categories h3 {
            color: #2c3e50;
        }
        .category-links {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .category-links a {
            background-color: #3498db;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 0.9em;
        }
        .category-links a:hover {
            background-color: #2980b9;
        }
        .intro {
            color: #555;
            margin-bottom: 20px;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <p>Find the best products at great prices</p>
    </header>
    <div class="container">
        <p class="intro">Search for products by name, category, or keyword below.</p>
        <form action="/search" method="get">
            <div class="search-box">
                <input type="text" name="query" placeholder="Search for products..." />
                <select name="category">
                    <option value="">All Categories</option>
                    <option value="Electronics">Electronics</option>
                    <option value="Shoes">Shoes</option>
                    <option value="Books">Books</option>
                    <option value="Kitchen">Kitchen</option>
                    <option value="Sports">Sports</option>
                    <option value="Home">Home</option>
                    <option value="Accessories">Accessories</option>
                </select>
                <button type="submit">Search</button>
            </div>
        </form>
        <div class="categories">
            <h3>Browse by Category:</h3>
            <div class="category-links">
                <a href="/search?category=Electronics">Electronics</a>
                <a href="/search?category=Shoes">Shoes</a>
                <a href="/search?category=Books">Books</a>
                <a href="/search?category=Kitchen">Kitchen</a>
                <a href="/search?category=Sports">Sports</a>
                <a href="/search?category=Home">Home</a>
                <a href="/search?category=Accessories">Accessories</a>
            </div>
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
    <title>Search Results - Online Shop</title>
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
        header h1 {
            margin: 0;
            font-size: 2em;
        }
        header a {
            color: #ecf0f1;
            text-decoration: none;
            font-size: 0.9em;
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
        }
        .search-bar input[type="text"] {
            flex: 1;
            padding: 10px;
            font-size: 1em;
            border: 2px solid #ccc;
            border-radius: 4px;
        }
        .search-bar select {
            padding: 10px;
            font-size: 1em;
            border: 2px solid #ccc;
            border-radius: 4px;
        }
        .search-bar button {
            padding: 10px 20px;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 1em;
            cursor: pointer;
        }
        .search-bar button:hover {
            background-color: #c0392b;
        }
        .results-info {
            color: #555;
            margin-bottom: 15px;
            font-size: 1em;
        }
        .results-info span {
            font-weight: bold;
            color: #2c3e50;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
        }
        .product-card h3 {
            margin: 0 0 8px 0;
            color: #2c3e50;
            font-size: 1.1em;
        }
        .product-card .category-badge {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .product-card p {
            color: #666;
            font-size: 0.9em;
            margin: 0 0 12px 0;
        }
        .product-card .price {
            font-size: 1.3em;
            font-weight: bold;
            color: #e74c3c;
        }
        .no-results {
            background: white;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: #555;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .no-results h3 {
            color: #2c3e50;
        }
        .no-results a {
            color: #3498db;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <a href="/">← Back to Home</a>
    </header>
    <div class="container">
        <div class="search-bar">
            <form action="/search" method="get">
                <input type="text" name="query" placeholder="Search for products..." value="{{ query }}" />
                <select name="category">
                    <option value="">All Categories</option>
                    <option value="Electronics" {% if category == 'Electronics' %}selected{% endif %}>Electronics</option>
                    <option value="Shoes" {% if category == 'Shoes' %}selected{% endif %}>Shoes</option>
                    <option value="Books" {% if category == 'Books' %}selected{% endif %}>Books</option>
                    <option value="Kitchen" {% if category == 'Kitchen' %}selected{% endif %}>Kitchen</option>
                    <option value="Sports" {% if category == 'Sports' %}selected{% endif %}>Sports</option>
                    <option value="Home" {% if category == 'Home' %}selected{% endif %}>Home</option>
                    <option value="Accessories" {% if category == 'Accessories' %}selected{% endif %}>Accessories</option>
                </select>
                <button type="submit">Search</button>
            </form>
        </div>

        {% if query or category %}
            <p class="results-info">
                Found <span>{{ products|length }}</span> result(s)
                {% if query %} for "<span>{{ query }}</span>"{% endif %}
                {% if category %} in category "<span>{{ category }}</span>"{% endif %}
            </p>
        {% else %}
            <p class="results-info">Showing all <span>{{ products|length }}</span> products</p>
        {% endif %}

        {% if products %}
            <div class="product-grid">
                {% for product in products %}
                <div class="product-card">
                    <h3>{{ product['name'] }}</h3>
                    <span class="category-badge">{{ product['category'] }}</span>
                    <p>{{ product['description'] }}</p>
                    <div class="price">${{ "%.2f"|format(product['price']) }}</div>
                </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="no-results">
                <h3>😕 No Products Found</h3>
                <p>We couldn't find any products matching your search.</p>
                <p>Try different keywords or <a href="/">browse all products</a>.</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

def search_products(query, category):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    sql = "SELECT * FROM products WHERE 1=1"
    params = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like_query = f"%{query}%"
        params.extend([like_query, like_query, like_query])

    if category:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY name ASC"

    cursor.execute(sql, params)
    results = cursor.fetchall()
    conn.close()
    return results

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "").strip()

    products = search_products(query, category)

    return render_template_string(
        RESULTS_TEMPLATE,
        products=products,
        query=query,
        category=category
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
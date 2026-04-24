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
        price REAL NOT NULL,
        keyword TEXT
    )''')
    c.execute("SELECT COUNT(*) FROM products")
    if c.fetchone()[0] == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone", 999.99, "phone mobile apple"),
            ("Samsung Galaxy S23", "Electronics", "Android flagship phone", 899.99, "phone mobile samsung android"),
            ("Sony WH-1000XM5", "Electronics", "Noise cancelling headphones", 349.99, "headphones audio sony"),
            ("Nike Air Max", "Shoes", "Comfortable running shoes", 129.99, "shoes running nike sport"),
            ("Adidas Ultraboost", "Shoes", "High performance running shoes", 179.99, "shoes running adidas sport"),
            ("Python Crash Course Book", "Books", "Beginner Python programming book", 29.99, "python programming book learning"),
            ("JavaScript: The Good Parts", "Books", "Classic JavaScript book", 24.99, "javascript programming book web"),
            ("Coffee Maker Deluxe", "Kitchen", "Brews perfect coffee every time", 79.99, "coffee kitchen appliance brew"),
            ("Blender Pro 3000", "Kitchen", "High speed blender for smoothies", 59.99, "blender kitchen smoothie"),
            ("Yoga Mat Premium", "Sports", "Non-slip yoga mat", 34.99, "yoga sports fitness mat"),
            ("Dumbbells Set 20kg", "Sports", "Adjustable dumbbell set", 89.99, "dumbbells weights fitness gym"),
            ("Office Chair Ergonomic", "Furniture", "Comfortable office chair", 249.99, "chair office ergonomic sitting"),
            ("Standing Desk", "Furniture", "Adjustable standing desk", 399.99, "desk standing office furniture"),
            ("Plant Pot Set", "Garden", "Set of 3 ceramic plant pots", 19.99, "plant pot garden ceramic"),
            ("Garden Tool Kit", "Garden", "Complete gardening tool set", 44.99, "garden tools kit outdoor"),
        ]
        c.executemany("INSERT INTO products (name, category, description, price, keyword) VALUES (?, ?, ?, ?, ?)", sample_products)
    conn.commit()
    conn.close()

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Small Online Shop</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .search-box {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        input[type="text"], select {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        label {
            font-weight: bold;
            color: #555;
        }
        .categories {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
            justify-content: center;
        }
        .category-btn {
            background-color: #007bff;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 14px;
            width: auto;
        }
        .category-btn:hover {
            background-color: #0056b3;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <h1>🛒 Small Online Shop</h1>
    <div class="search-box">
        <form action="/search" method="get">
            <label for="query">Search Products:</label>
            <input type="text" id="query" name="query" placeholder="e.g. phone, shoes, python book..." value="{{ query }}">

            <label for="category">Filter by Category (optional):</label>
            <select id="category" name="category">
                <option value="">All Categories</option>
                {% for cat in categories %}
                <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                {% endfor %}
            </select>

            <button type="submit">🔍 Search</button>
        </form>
    </div>

    <h3 style="text-align:center;">Browse by Category:</h3>
    <div class="categories">
        {% for cat in categories %}
        <a href="/search?category={{ cat }}" class="category-btn">{{ cat }}</a>
        {% endfor %}
    </div>

    <div class="footer">
        <p>Simple Online Shop &copy; 2024</p>
    </div>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results - Small Online Shop</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .search-again {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .search-again form {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-again input[type="text"] {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 15px;
            min-width: 200px;
        }
        .search-again select {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 15px;
        }
        .search-again button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 15px;
            cursor: pointer;
        }
        .search-again button:hover {
            background-color: #45a049;
        }
        .result-count {
            color: #555;
            margin-bottom: 15px;
            font-size: 15px;
        }
        .product-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .product-info h3 {
            margin: 0 0 5px 0;
            color: #222;
        }
        .product-info p {
            margin: 0;
            color: #666;
            font-size: 14px;
        }
        .product-info .category-tag {
            display: inline-block;
            background-color: #e0f0ff;
            color: #007bff;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            margin-top: 6px;
        }
        .price {
            font-size: 22px;
            font-weight: bold;
            color: #4CAF50;
            white-space: nowrap;
        }
        .no-results {
            background: white;
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            color: #888;
        }
        .no-results h2 {
            color: #aaa;
        }
        .back-link {
            text-align: center;
            margin-top: 20px;
        }
        .back-link a {
            color: #007bff;
            text-decoration: none;
        }
        .back-link a:hover {
            text-decoration: underline;
        }
        .add-btn {
            background-color: #007bff;
            color: white;
            padding: 8px 18px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 5px;
        }
        .add-btn:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <h1>🛒 Search Results</h1>

    <div class="search-again">
        <form action="/search" method="get">
            <input type="text" name="query" placeholder="Search again..." value="{{ query }}">
            <select name="category">
                <option value="">All Categories</option>
                {% for cat in categories %}
                <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                {% endfor %}
            </select>
            <button type="submit">🔍 Search</button>
        </form>
    </div>

    {% if query or selected_category %}
        <p class="result-count">
            Found <strong>{{ products|length }}</strong> result(s)
            {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
            {% if selected_category %} in category "<strong>{{ selected_category }}</strong>"{% endif %}
        </p>
    {% endif %}

    {% if products %}
        {% for product in products %}
        <div class="product-card">
            <div class="product-info">
                <h3>{{ product.name }}</h3>
                <p>{{ product.description }}</p>
                <span class="category-tag">{{ product.category }}</span>
                <br>
                <button class="add-btn">🛒 Add to Cart</button>
            </div>
            <div class="price">${{ "%.2f"|format(product.price) }}</div>
        </div>
        {% endfor %}
    {% elif query or selected_category %}
        <div class="no-results">
            <h2>😕 No Products Found</h2>
            <p>Try different keywords or browse all categories.</p>
        </div>
    {% else %}
        <div class="no-results">
            <h2>All Products</h2>
        </div>
    {% endif %}

    <div class="back-link">
        <a href="/">← Back to Home</a>
    </div>
</body>
</html>
"""

def get_categories():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT category FROM products ORDER BY category")
    cats = [row[0] for row in c.fetchall()]
    conn.close()
    return cats

def search_products(query, category):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if query and category:
        like_query = f"%{query}%"
        c.execute("""
            SELECT * FROM products
            WHERE category = ?
            AND (name LIKE ? OR description LIKE ? OR keyword LIKE ?)
            ORDER BY name
        """, (category, like_query, like_query, like_query))
    elif query:
        like_query = f"%{query}%"
        c.execute("""
            SELECT * FROM products
            WHERE name LIKE ? OR description LIKE ? OR keyword LIKE ? OR category LIKE ?
            ORDER BY name
        """, (like_query, like_query, like_query, like_query))
    elif category:
        c.execute("SELECT * FROM products WHERE category = ? ORDER BY name", (category,))
    else:
        c.execute("SELECT * FROM products ORDER BY name")

    results = c.fetchall()
    conn.close()
    return results

@app.route("/")
def home():
    categories = get_categories()
    return render_template_string(
        HOME_HTML,
        categories=categories,
        query="",
        selected_category=""
    )

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "").strip()
    categories = get_categories()
    products = search_products(query, category)
    return render_template_string(
        RESULTS_HTML,
        products=products,
        query=query,
        selected_category=category,
        categories=categories
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
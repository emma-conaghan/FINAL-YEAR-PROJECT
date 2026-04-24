from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)
DB_NAME = "shop.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL
        )
    ''')
    c.execute("SELECT COUNT(*) FROM products")
    count = c.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with A15 chip", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Android smartphone with great camera", 849.99),
            ("Nike Air Max", "Shoes", "Comfortable running shoes", 129.99),
            ("Adidas Ultraboost", "Shoes", "High performance running shoes", 179.99),
            ("Python Programming Book", "Books", "Learn Python from scratch", 39.99),
            ("JavaScript Guide", "Books", "Complete guide to JavaScript", 34.99),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker", 59.99),
            ("Blender Pro", "Kitchen", "High speed blender for smoothies", 79.99),
            ("Yoga Mat", "Sports", "Non-slip yoga and exercise mat", 24.99),
            ("Dumbbell Set", "Sports", "Adjustable dumbbell set for home gym", 149.99),
            ("Laptop Stand", "Office", "Ergonomic aluminum laptop stand", 44.99),
            ("Wireless Mouse", "Office", "Bluetooth wireless mouse", 29.99),
            ("Headphones", "Electronics", "Noise cancelling over-ear headphones", 199.99),
            ("Backpack", "Accessories", "Waterproof travel backpack", 69.99),
            ("Sunglasses", "Accessories", "UV400 protection sunglasses", 49.99),
        ]
        c.executemany("INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)", sample_products)
    conn.commit()
    conn.close()

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Small Online Shop</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
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
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 40px;
        }
        input[type="text"] {
            width: 60%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        select {
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>🛒 Small Online Shop</h1>
    <p class="subtitle">Find the products you are looking for!</p>
    <div class="search-box">
        <h2>Search Products</h2>
        <form action="/search" method="get">
            <input type="text" name="query" placeholder="Search by name or keyword..." />
            <select name="category">
                <option value="">All Categories</option>
                <option value="Electronics">Electronics</option>
                <option value="Shoes">Shoes</option>
                <option value="Books">Books</option>
                <option value="Kitchen">Kitchen</option>
                <option value="Sports">Sports</option>
                <option value="Office">Office</option>
                <option value="Accessories">Accessories</option>
            </select>
            <button type="submit">Search</button>
        </form>
    </div>
</body>
</html>
'''

RESULTS_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Search Results - Small Online Shop</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .search-box {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 20px;
        }
        input[type="text"] {
            width: 50%;
            padding: 10px;
            font-size: 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        select {
            padding: 10px;
            font-size: 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        button {
            padding: 10px 20px;
            font-size: 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .back-link {
            display: inline-block;
            margin-bottom: 15px;
            color: #4CAF50;
            text-decoration: none;
            font-size: 15px;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        .results-info {
            color: #555;
            margin-bottom: 15px;
            font-size: 15px;
        }
        .product-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .product-info h3 {
            margin: 0 0 5px 0;
            color: #333;
        }
        .product-info .category {
            display: inline-block;
            background-color: #e0f0e0;
            color: #4CAF50;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 13px;
            margin-bottom: 8px;
        }
        .product-info .description {
            color: #666;
            font-size: 14px;
            margin: 0;
        }
        .product-price {
            font-size: 22px;
            font-weight: bold;
            color: #e44d26;
            white-space: nowrap;
        }
        .no-results {
            background: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
            color: #666;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <h1>🛒 Small Online Shop</h1>

    <div class="search-box">
        <form action="/search" method="get">
            <input type="text" name="query" placeholder="Search by name or keyword..." value="{{ query }}" />
            <select name="category">
                <option value="">All Categories</option>
                <option value="Electronics" {% if category == "Electronics" %}selected{% endif %}>Electronics</option>
                <option value="Shoes" {% if category == "Shoes" %}selected{% endif %}>Shoes</option>
                <option value="Books" {% if category == "Books" %}selected{% endif %}>Books</option>
                <option value="Kitchen" {% if category == "Kitchen" %}selected{% endif %}>Kitchen</option>
                <option value="Sports" {% if category == "Sports" %}selected{% endif %}>Sports</option>
                <option value="Office" {% if category == "Office" %}selected{% endif %}>Office</option>
                <option value="Accessories" {% if category == "Accessories" %}selected{% endif %}>Accessories</option>
            </select>
            <button type="submit">Search</button>
        </form>
    </div>

    <a class="back-link" href="/">← Back to Home</a>

    <p class="results-info">
        Found <strong>{{ results|length }}</strong> result(s)
        {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
        {% if category %} in category "<strong>{{ category }}</strong>"{% endif %}
    </p>

    {% if results %}
        {% for product in results %}
        <div class="product-card">
            <div class="product-info">
                <h3>{{ product[1] }}</h3>
                <span class="category">{{ product[2] }}</span>
                <p class="description">{{ product[3] }}</p>
            </div>
            <div class="product-price">${{ "%.2f"|format(product[4]) }}</div>
        </div>
        {% endfor %}
    {% else %}
        <div class="no-results">
            <h3>No products found</h3>
            <p>Try a different search term or browse all categories.</p>
        </div>
    {% endif %}

</body>
</html>
'''

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "").strip()

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    if query and category:
        like_query = "%" + query + "%"
        c.execute(
            "SELECT * FROM products WHERE category = ? AND (name LIKE ? OR description LIKE ?)",
            (category, like_query, like_query)
        )
    elif query:
        like_query = "%" + query + "%"
        c.execute(
            "SELECT * FROM products WHERE name LIKE ? OR description LIKE ?",
            (like_query, like_query)
        )
    elif category:
        c.execute(
            "SELECT * FROM products WHERE category = ?",
            (category,)
        )
    else:
        c.execute("SELECT * FROM products")

    results = c.fetchall()
    conn.close()

    return render_template_string(RESULTS_TEMPLATE, results=results, query=query, category=category)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
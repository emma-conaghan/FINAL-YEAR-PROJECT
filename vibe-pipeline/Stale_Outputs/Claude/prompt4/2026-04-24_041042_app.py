from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)

DB_NAME = "shop.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    if count == 0:
        sample_products = [
            ("Apple iPhone 14", "Electronics", "Latest Apple smartphone with great camera", 999.99),
            ("Samsung Galaxy S23", "Electronics", "Android flagship phone with AMOLED display", 899.99),
            ("Sony Headphones", "Electronics", "Noise cancelling wireless headphones", 299.99),
            ("Nike Running Shoes", "Footwear", "Comfortable shoes for everyday running", 120.00),
            ("Adidas Sneakers", "Footwear", "Classic style sneakers for casual wear", 90.00),
            ("Levi's Jeans", "Clothing", "Classic blue denim jeans", 60.00),
            ("Cotton T-Shirt", "Clothing", "Soft and comfortable everyday t-shirt", 25.00),
            ("Harry Potter Book", "Books", "The complete Harry Potter series collection", 45.00),
            ("Python Programming Book", "Books", "Learn Python from scratch beginner guide", 35.00),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker with timer", 55.00),
            ("Blender", "Kitchen", "High speed blender for smoothies and shakes", 80.00),
            ("Yoga Mat", "Sports", "Non slip yoga mat for home workout", 30.00),
            ("Dumbbells Set", "Sports", "Adjustable dumbbells set for strength training", 150.00),
            ("Laptop Stand", "Electronics", "Ergonomic aluminum laptop stand", 40.00),
            ("Mechanical Keyboard", "Electronics", "RGB mechanical keyboard for gaming and typing", 120.00),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)",
            sample_products
        )
    conn.commit()
    conn.close()

def search_products(query):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    search_term = f"%{query}%"
    cursor.execute('''
        SELECT id, name, category, description, price FROM products
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
    ''', (search_term, search_term, search_term))
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

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Online Shop</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        header {
            background-color: #2c3e50;
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
            margin: 40px auto;
            padding: 0 20px;
        }
        .search-box {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 30px;
        }
        .search-box h2 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .search-box input[type="text"] {
            width: 60%;
            padding: 12px 15px;
            font-size: 16px;
            border: 2px solid #ccc;
            border-radius: 5px;
            outline: none;
        }
        .search-box input[type="text"]:focus {
            border-color: #2980b9;
        }
        .search-box button {
            padding: 12px 25px;
            font-size: 16px;
            background-color: #2980b9;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-left: 10px;
        }
        .search-box button:hover {
            background-color: #1a6fa3;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .product-card h3 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 16px;
        }
        .product-card .category {
            display: inline-block;
            background-color: #eaf3fb;
            color: #2980b9;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .product-card .description {
            color: #666;
            font-size: 13px;
            margin-bottom: 15px;
        }
        .product-card .price {
            font-size: 20px;
            font-weight: bold;
            color: #27ae60;
        }
        .section-title {
            font-size: 20px;
            color: #2c3e50;
            margin-bottom: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ My Online Shop</h1>
    </header>
    <div class="container">
        <div class="search-box">
            <h2>Find What You Need</h2>
            <form method="GET" action="/search">
                <input type="text" name="q" placeholder="Search by name, category, or keyword..." required>
                <button type="submit">Search</button>
            </form>
        </div>
        <div class="section-title">All Products</div>
        <div class="products-grid">
            {% for product in products %}
            <div class="product-card">
                <h3>{{ product[1] }}</h3>
                <span class="category">{{ product[2] }}</span>
                <p class="description">{{ product[3] }}</p>
                <div class="price">${{ "%.2f"|format(product[4]) }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
'''

SEARCH_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results - Online Shop</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        header {
            background-color: #2c3e50;
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
            margin: 40px auto;
            padding: 0 20px;
        }
        .search-box {
            background: white;
            padding: 20px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .search-box form {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .search-box input[type="text"] {
            flex: 1;
            padding: 12px 15px;
            font-size: 16px;
            border: 2px solid #ccc;
            border-radius: 5px;
            outline: none;
        }
        .search-box input[type="text"]:focus {
            border-color: #2980b9;
        }
        .search-box button {
            padding: 12px 25px;
            font-size: 16px;
            background-color: #2980b9;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .search-box button:hover {
            background-color: #1a6fa3;
        }
        .back-link {
            display: inline-block;
            margin-bottom: 15px;
            color: #2980b9;
            text-decoration: none;
            font-size: 14px;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        .results-info {
            color: #555;
            margin-bottom: 20px;
            font-size: 15px;
        }
        .results-info span {
            font-weight: bold;
            color: #2c3e50;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .product-card h3 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 16px;
        }
        .product-card .category {
            display: inline-block;
            background-color: #eaf3fb;
            color: #2980b9;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .product-card .description {
            color: #666;
            font-size: 13px;
            margin-bottom: 15px;
        }
        .product-card .price {
            font-size: 20px;
            font-weight: bold;
            color: #27ae60;
        }
        .no-results {
            background: white;
            padding: 40px;
            text-align: center;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            color: #666;
        }
        .no-results h3 {
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ My Online Shop</h1>
    </header>
    <div class="container">
        <a href="/" class="back-link">← Back to All Products</a>
        <div class="search-box">
            <form method="GET" action="/search">
                <input type="text" name="q" value="{{ query }}" placeholder="Search by name, category, or keyword..." required>
                <button type="submit">Search</button>
            </form>
        </div>
        <div class="results-info">
            Showing <span>{{ results|length }}</span> result(s) for "<span>{{ query }}</span>"
        </div>
        {% if results %}
        <div class="products-grid">
            {% for product in results %}
            <div class="product-card">
                <h3>{{ product[1] }}</h3>
                <span class="category">{{ product[2] }}</span>
                <p class="description">{{ product[3] }}</p>
                <div class="price">${{ "%.2f"|format(product[4]) }}</div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <h3>No products found</h3>
            <p>Try searching with a different keyword or browse all products.</p>
            <a href="/">View all products</a>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route("/")
def home():
    products = get_all_products()
    return render_template_string(HOME_TEMPLATE, products=products)

@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return home()
    results = search_products(query)
    return render_template_string(SEARCH_TEMPLATE, results=results, query=query)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
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
            ("Samsung Galaxy S23", "Electronics", "Flagship Android phone with great camera", 849.99),
            ("Sony Headphones WH-1000XM5", "Electronics", "Noise cancelling wireless headphones", 349.99),
            ("Nike Running Shoes", "Footwear", "Lightweight shoes perfect for running", 89.99),
            ("Adidas Ultraboost", "Footwear", "Comfortable sneakers for everyday use", 120.00),
            ("Levi's Jeans", "Clothing", "Classic blue denim jeans", 59.99),
            ("Cotton T-Shirt", "Clothing", "Basic comfortable cotton t-shirt", 19.99),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for home workouts", 29.99),
            ("Dumbbell Set", "Sports", "Adjustable dumbbell set for strength training", 149.99),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker with timer", 49.99),
            ("Blender", "Kitchen", "High speed blender for smoothies and soups", 39.99),
            ("Python Programming Book", "Books", "Learn Python from scratch beginner guide", 34.99),
            ("Science Fiction Novel", "Books", "Exciting space adventure story", 14.99),
            ("Desk Lamp", "Home", "LED desk lamp with adjustable brightness", 24.99),
            ("Throw Pillow", "Home", "Soft decorative pillow for sofa or bed", 12.99),
        ]
        c.executemany("INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)", sample_products)
    conn.commit()
    conn.close()

def search_products(query):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    search_term = f"%{query}%"
    c.execute('''
        SELECT id, name, category, description, price
        FROM products
        WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
    ''', (search_term, search_term, search_term))
    results = c.fetchall()
    conn.close()
    return results

def get_all_products():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, category, description, price FROM products")
    results = c.fetchall()
    conn.close()
    return results

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Small Online Shop</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; color: #333; }
        header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        header h1 { font-size: 2em; margin-bottom: 5px; }
        header p { font-size: 1em; color: #bdc3c7; }
        .search-section {
            background: white;
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .search-section form {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .search-section input[type="text"] {
            padding: 12px 20px;
            font-size: 1em;
            border: 2px solid #2c3e50;
            border-radius: 25px;
            width: 400px;
            max-width: 100%;
            outline: none;
        }
        .search-section input[type="text"]:focus {
            border-color: #3498db;
        }
        .search-section button {
            padding: 12px 30px;
            font-size: 1em;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
        }
        .search-section button:hover { background: #2980b9; }
        .categories {
            padding: 20px;
            text-align: center;
        }
        .categories h2 { margin-bottom: 15px; color: #2c3e50; }
        .category-buttons {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
        }
        .category-btn {
            padding: 8px 20px;
            background: #ecf0f1;
            border: 2px solid #bdc3c7;
            border-radius: 20px;
            cursor: pointer;
            text-decoration: none;
            color: #2c3e50;
            font-size: 0.9em;
        }
        .category-btn:hover { background: #2c3e50; color: white; border-color: #2c3e50; }
        .products-section { padding: 20px 40px; }
        .products-section h2 { margin-bottom: 20px; color: #2c3e50; }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover { transform: translateY(-4px); }
        .product-card .category-tag {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.75em;
            margin-bottom: 10px;
        }
        .product-card h3 { font-size: 1em; margin-bottom: 8px; }
        .product-card p { font-size: 0.85em; color: #777; margin-bottom: 10px; }
        .product-card .price { font-size: 1.2em; font-weight: bold; color: #27ae60; }
        .no-results { text-align: center; padding: 40px; color: #888; font-size: 1.1em; }
        footer {
            text-align: center;
            padding: 20px;
            background: #2c3e50;
            color: #bdc3c7;
            margin-top: 40px;
            font-size: 0.85em;
        }
    </style>
</head>
<body>

<header>
    <h1>🛒 Small Online Shop</h1>
    <p>Find great products at great prices</p>
</header>

<div class="search-section">
    <h2>Search for Products</h2>
    <form action="/search" method="GET">
        <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query or '' }}">
        <button type="submit">🔍 Search</button>
    </form>
</div>

<div class="categories">
    <h2>Browse by Category</h2>
    <div class="category-buttons">
        {% for cat in categories %}
        <a href="/search?q={{ cat }}" class="category-btn">{{ cat }}</a>
        {% endfor %}
    </div>
</div>

<div class="products-section">
    {% if search_performed %}
        <h2>Search Results for "{{ query }}" ({{ products|length }} found)</h2>
    {% else %}
        <h2>All Products</h2>
    {% endif %}

    {% if products %}
    <div class="product-grid">
        {% for product in products %}
        <div class="product-card">
            <span class="category-tag">{{ product[2] }}</span>
            <h3>{{ product[1] }}</h3>
            <p>{{ product[3] }}</p>
            <div class="price">${{ "%.2f"|format(product[4]) }}</div>
        </div>
        {% endfor %}
    </div>
    {% else %}
        <div class="no-results">
            😕 No products found for "{{ query }}". Try a different search term.
        </div>
    {% endif %}
</div>

<footer>
    <p>&copy; 2024 Small Online Shop. All rights reserved.</p>
</footer>

</body>
</html>
'''

@app.route("/")
def home():
    products = get_all_products()
    categories = sorted(set(p[2] for p in products))
    return render_template_string(
        HOME_TEMPLATE,
        products=products,
        categories=categories,
        query=None,
        search_performed=False
    )

@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    all_products = get_all_products()
    categories = sorted(set(p[2] for p in all_products))

    if query:
        products = search_products(query)
        search_performed = True
    else:
        products = all_products
        search_performed = False

    return render_template_string(
        HOME_TEMPLATE,
        products=products,
        categories=categories,
        query=query,
        search_performed=search_performed
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
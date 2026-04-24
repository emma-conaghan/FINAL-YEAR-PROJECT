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
            ("Samsung Galaxy S23", "Electronics", "Android smartphone with powerful processor", 849.99),
            ("Nike Running Shoes", "Footwear", "Comfortable shoes for running and jogging", 89.99),
            ("Adidas Sneakers", "Footwear", "Stylish sneakers for everyday wear", 69.99),
            ("Harry Potter Book Set", "Books", "Complete Harry Potter series box set", 49.99),
            ("Python Programming Book", "Books", "Learn Python programming from scratch", 29.99),
            ("Coffee Maker", "Kitchen", "Automatic drip coffee maker with timer", 39.99),
            ("Blender Pro", "Kitchen", "High speed blender for smoothies and more", 59.99),
            ("Yoga Mat", "Sports", "Non-slip yoga mat for home workouts", 24.99),
            ("Dumbbells Set", "Sports", "Adjustable dumbbells for strength training", 149.99),
            ("Laptop Stand", "Electronics", "Ergonomic aluminum laptop stand", 34.99),
            ("Wireless Headphones", "Electronics", "Noise cancelling bluetooth headphones", 199.99),
            ("Cooking Pan Set", "Kitchen", "Non-stick cooking pans set of 3", 44.99),
            ("Running Shorts", "Footwear", "Breathable running shorts for athletes", 19.99),
            ("Science Fiction Novel", "Books", "Exciting sci-fi adventure story", 14.99),
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
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .search-section {
            text-align: center;
            margin-bottom: 30px;
        }
        .search-section h2 {
            color: #2c3e50;
        }
        .search-form {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }
        .search-form input[type="text"] {
            width: 60%;
            padding: 12px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 5px;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #2c3e50;
        }
        .search-form select {
            padding: 10px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 5px;
            outline: none;
        }
        .search-form button {
            padding: 12px 30px;
            font-size: 1em;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .search-form button:hover {
            background-color: #c0392b;
        }
        .categories {
            text-align: center;
            margin-top: 20px;
        }
        .categories a {
            display: inline-block;
            margin: 5px;
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .categories a:hover {
            background-color: #2980b9;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .product-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
            transition: box-shadow 0.2s;
        }
        .product-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .product-card h3 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 1em;
        }
        .product-card .category-badge {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-bottom: 8px;
        }
        .product-card .description {
            font-size: 0.9em;
            color: #666;
            margin: 8px 0;
        }
        .product-card .price {
            font-size: 1.2em;
            color: #e74c3c;
            font-weight: bold;
        }
        .results-header {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .no-results {
            text-align: center;
            color: #999;
            font-size: 1.1em;
            margin: 40px 0;
        }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #3498db;
            text-decoration: none;
            font-size: 0.95em;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <p>Find great products at amazing prices</p>
    </header>
    <div class="container">
        <div class="search-section">
            <h2>Search Products</h2>
            <form class="search-form" action="/search" method="get">
                <input type="text" name="query" placeholder="Search by name, category, or keyword..." value="{{ query }}">
                <select name="category">
                    <option value="">All Categories</option>
                    <option value="Electronics" {% if selected_category == 'Electronics' %}selected{% endif %}>Electronics</option>
                    <option value="Footwear" {% if selected_category == 'Footwear' %}selected{% endif %}>Footwear</option>
                    <option value="Books" {% if selected_category == 'Books' %}selected{% endif %}>Books</option>
                    <option value="Kitchen" {% if selected_category == 'Kitchen' %}selected{% endif %}>Kitchen</option>
                    <option value="Sports" {% if selected_category == 'Sports' %}selected{% endif %}>Sports</option>
                </select>
                <button type="submit">🔍 Search</button>
            </form>
        </div>
        <div class="categories">
            <p><strong>Browse by category:</strong></p>
            <a href="/search?category=Electronics">Electronics</a>
            <a href="/search?category=Footwear">Footwear</a>
            <a href="/search?category=Books">Books</a>
            <a href="/search?category=Kitchen">Kitchen</a>
            <a href="/search?category=Sports">Sports</a>
        </div>
    </div>
    <footer>
        <p>&copy; 2024 My Online Shop. All rights reserved.</p>
    </footer>
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
        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .search-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .search-bar input[type="text"] {
            flex: 1;
            padding: 10px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 5px;
            outline: none;
            min-width: 200px;
        }
        .search-bar input[type="text"]:focus {
            border-color: #2c3e50;
        }
        .search-bar select {
            padding: 10px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 5px;
            outline: none;
        }
        .search-bar button {
            padding: 10px 20px;
            font-size: 1em;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .search-bar button:hover {
            background-color: #c0392b;
        }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #3498db;
            text-decoration: none;
            font-size: 0.95em;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        .results-header {
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .product-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
            transition: box-shadow 0.2s;
        }
        .product-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .product-card h3 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 1em;
        }
        .product-card .category-badge {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-bottom: 8px;
        }
        .product-card .description {
            font-size: 0.9em;
            color: #666;
            margin: 8px 0;
        }
        .product-card .price {
            font-size: 1.2em;
            color: #e74c3c;
            font-weight: bold;
        }
        .no-results {
            text-align: center;
            color: #999;
            font-size: 1.1em;
            margin: 40px 0;
        }
        .no-results span {
            font-size: 3em;
            display: block;
            margin-bottom: 10px;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 My Online Shop</h1>
        <p>Find great products at amazing prices</p>
    </header>
    <div class="container">
        <a class="back-link" href="/">← Back to Home</a>
        <form class="search-bar" action="/search" method="get">
            <input type="text" name="query" placeholder="Search products..." value="{{ query }}">
            <select name="category">
                <option value="">All Categories</option>
                <option value="Electronics" {% if selected_category == 'Electronics' %}selected{% endif %}>Electronics</option>
                <option value="Footwear" {% if selected_category == 'Footwear' %}selected{% endif %}>Footwear</option>
                <option value="Books" {% if selected_category == 'Books' %}selected{% endif %}>Books</option>
                <option value="Kitchen" {% if selected_category == 'Kitchen' %}selected{% endif %}>Kitchen</option>
                <option value="Sports" {% if selected_category == 'Sports' %}selected{% endif %}>Sports</option>
            </select>
            <button type="submit">🔍 Search</button>
        </form>
        <div class="results-header">
            {% if query or selected_category %}
                <h2>
                    Search Results
                    {% if query %}for "<strong>{{ query }}</strong>"{% endif %}
                    {% if selected_category %}in
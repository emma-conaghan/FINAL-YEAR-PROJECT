from flask import Flask, request, render_template_string
import sqlite3

app = Flask(__name__)

DATABASE = 'shop.db'


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            keywords TEXT
        )
    ''')
    
    cursor.execute('SELECT COUNT(*) FROM products')
    count = cursor.fetchone()[0]
    
    if count == 0:
        sample_products = [
            ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 29.99, 'mouse wireless computer peripheral'),
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 79.99, 'keyboard mechanical gaming RGB computer'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 45.99, 'USB hub adapter dongle computer'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 35.99, 'laptop stand desk ergonomic'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 59.99, 'webcam camera video streaming'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 24.99, 'lamp light LED desk office'),
            ('Notebook Journal', 'Stationery', 'Hardcover lined notebook, 200 pages', 12.99, 'notebook journal writing paper'),
            ('Ballpoint Pen Set', 'Stationery', 'Set of 10 premium ballpoint pens', 8.99, 'pen writing stationery office'),
            ('Monitor Arm', 'Accessories', 'Single monitor arm mount for desks', 49.99, 'monitor arm mount desk ergonomic'),
            ('Wireless Charger', 'Electronics', 'Fast wireless charging pad for smartphones', 19.99, 'charger wireless phone charging'),
            ('Noise Cancelling Headphones', 'Electronics', 'Over-ear noise cancelling Bluetooth headphones', 129.99, 'headphones audio music bluetooth noise cancelling'),
            ('Coffee Mug', 'Home Office', 'Ceramic coffee mug with funny coding quote', 14.99, 'mug coffee cup office gift'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 16.99, 'mousepad mouse pad desk gaming'),
            ('Portable SSD 1TB', 'Electronics', 'External portable SSD with 1TB storage', 89.99, 'SSD storage drive portable external'),
            ('Blue Light Glasses', 'Accessories', 'Computer glasses with blue light filter', 22.99, 'glasses blue light filter computer eye'),
            ('Standing Desk Mat', 'Home Office', 'Anti-fatigue standing desk mat', 39.99, 'mat standing desk ergonomic comfort'),
            ('Sticky Notes Pack', 'Stationery', 'Colorful sticky notes, 12 pads', 6.99, 'sticky notes post-it stationery office'),
            ('Cable Management Kit', 'Accessories', 'Cable clips and sleeves for desk organization', 11.99, 'cable management organizer desk tidy'),
            ('Smartphone Stand', 'Accessories', 'Adjustable phone stand for desk', 13.99, 'phone stand holder desk smartphone'),
            ('Ergonomic Chair Cushion', 'Home Office', 'Memory foam seat cushion for office chairs', 34.99, 'cushion chair ergonomic comfort seat'),
        ]
        
        cursor.executemany(
            'INSERT INTO products (name, category, description, price, keywords) VALUES (?, ?, ?, ?, ?)',
            sample_products
        )
    
    conn.commit()
    conn.close()


HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Online Shop</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }
        header h1 {
            font-size: 2.2em;
            margin-bottom: 8px;
        }
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-box {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin: 30px auto;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .search-box h2 {
            margin-bottom: 20px;
            color: #555;
        }
        .search-form {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            padding: 12px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            width: 400px;
            max-width: 100%;
            outline: none;
            transition: border-color 0.3s;
        }
        .search-form input[type="text"]:focus {
            border-color: #667eea;
        }
        .search-form button {
            padding: 12px 30px;
            font-size: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: opacity 0.3s;
        }
        .search-form button:hover {
            opacity: 0.9;
        }
        .categories {
            margin: 30px auto;
            text-align: center;
        }
        .categories h3 {
            margin-bottom: 15px;
            color: #555;
        }
        .category-links {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .category-links a {
            padding: 8px 20px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 20px;
            border: 2px solid #667eea;
            transition: all 0.3s;
            font-weight: 500;
        }
        .category-links a:hover {
            background: #667eea;
            color: white;
        }
        .results-info {
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        }
        .product-card h3 {
            color: #333;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        .product-card .category {
            display: inline-block;
            background: #eef0fb;
            color: #667eea;
            padding: 3px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            margin-bottom: 10px;
        }
        .product-card .description {
            color: #666;
            font-size: 0.95em;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 1.3em;
            font-weight: bold;
            color: #2d8f2d;
        }
        .product-card .add-to-cart {
            display: inline-block;
            margin-top: 10px;
            padding: 8px 18px;
            background: #2d8f2d;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            text-decoration: none;
        }
        .product-card .add-to-cart:hover {
            background: #247a24;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #888;
        }
        .no-results h3 {
            font-size: 1.3em;
            margin-bottom: 10px;
        }
        .back-link {
            display: inline-block;
            margin: 15px 0;
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        .all-products-link {
            margin-top: 15px;
        }
        .all-products-link a {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        .all-products-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Simple Online Shop</h1>
        <p>Find great products at great prices</p>
    </header>
    <div class="container">
        <div class="search-box">
            <h2>Search for Products</h2>
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query or '' }}">
                <button type="submit">🔍 Search</button>
            </form>
            <div class="all-products-link" style="margin-top: 15px;">
                <a href="/search?q=">Browse All Products</a>
            </div>
        </div>
        
        <div class="categories">
            <h3>Browse by Category</h3>
            <div class="category-links">
                {% for cat in categories %}
                <a href="/search?q={{ cat }}">{{ cat }}</a>
                {% endfor %}
            </div>
        </div>
    </div>
</body>
</html>
'''

RESULTS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results - Simple Online Shop</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        header h1 {
            font-size: 1.8em;
            margin-bottom: 5px;
        }
        header h1 a {
            color: white;
            text-decoration: none;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-box {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 20px auto;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .search-form {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            padding: 12px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            width: 400px;
            max-width: 100%;
            outline: none;
            transition: border-color 0.3s;
        }
        .search-form input[type="text"]:focus {
            border-color: #667eea;
        }
        .search-form button {
            padding: 12px 30px;
            font-size: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: opacity 0.3s;
        }
        .search-form button:hover {
            opacity: 0.9;
        }
        .results-info {
            margin: 20px 0 10px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 20
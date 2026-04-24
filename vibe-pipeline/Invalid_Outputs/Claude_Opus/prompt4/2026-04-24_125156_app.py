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
            image_url TEXT
        )
    ''')
    
    cursor.execute('SELECT COUNT(*) FROM products')
    count = cursor.fetchone()[0]
    
    if count == 0:
        sample_products = [
            ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 19.99, '🖱️'),
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 49.99, '⌨️'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI output', 29.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 35.99, '💻'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 39.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 24.99, '💡'),
            ('Notebook Set', 'Stationery', 'Pack of 3 ruled notebooks, A5 size', 9.99, '📓'),
            ('Gel Pen Pack', 'Stationery', 'Set of 12 colorful gel pens', 7.99, '🖊️'),
            ('Monitor Riser', 'Accessories', 'Wooden monitor riser with storage', 42.99, '🖥️'),
            ('Bluetooth Speaker', 'Electronics', 'Portable Bluetooth speaker, waterproof', 34.99, '🔊'),
            ('Phone Stand', 'Accessories', 'Adjustable phone stand for desk', 12.99, '📱'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 14.99, '🎯'),
            ('Cable Organizer', 'Home Office', 'Silicone cable management clips', 6.99, '🔗'),
            ('Sticky Notes', 'Stationery', 'Colorful sticky notes, 500 sheets', 5.99, '📝'),
            ('Desk Organizer', 'Home Office', 'Bamboo desk organizer with compartments', 22.99, '🗂️'),
            ('Headphone Stand', 'Accessories', 'Aluminum headphone stand', 18.99, '🎧'),
            ('Whiteboard', 'Home Office', 'Magnetic dry-erase whiteboard, 24x18 inches', 27.99, '📋'),
            ('Ergonomic Chair Cushion', 'Home Office', 'Memory foam seat cushion for office chair', 31.99, '🪑'),
            ('Drawing Tablet', 'Electronics', 'Graphics drawing tablet for digital art', 59.99, '🎨'),
            ('Book Light', 'Accessories', 'Rechargeable clip-on book light', 11.99, '📖'),
        ]
        
        cursor.executemany(
            'INSERT INTO products (name, category, description, price, image_url) VALUES (?, ?, ?, ?, ?)',
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 20px 40px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: #333;
            font-size: 28px;
        }
        .header h1 span {
            color: #667eea;
        }
        .nav-links {
            margin-top: 8px;
        }
        .nav-links a {
            color: #666;
            text-decoration: none;
            margin-right: 20px;
            font-size: 14px;
        }
        .nav-links a:hover {
            color: #667eea;
        }
        .hero {
            text-align: center;
            padding: 60px 20px 40px;
            color: white;
        }
        .hero h2 {
            font-size: 42px;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .hero p {
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        .search-container {
            max-width: 600px;
            margin: 0 auto;
            position: relative;
        }
        .search-form {
            display: flex;
            background: white;
            border-radius: 50px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .search-form input[type="text"] {
            flex: 1;
            padding: 18px 30px;
            border: none;
            font-size: 16px;
            outline: none;
        }
        .search-form button {
            padding: 18px 35px;
            background: #667eea;
            color: white;
            border: none;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .search-form button:hover {
            background: #5a6fd6;
        }
        .categories {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .category-btn {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 8px 20px;
            border-radius: 25px;
            cursor: pointer;
            text-decoration: none;
            font-size: 14px;
            transition: all 0.3s;
        }
        .category-btn:hover {
            background: rgba(255,255,255,0.3);
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .section-title {
            color: white;
            font-size: 24px;
            margin-bottom: 20px;
            text-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        .product-icon {
            font-size: 48px;
            text-align: center;
            margin-bottom: 15px;
        }
        .product-name {
            font-size: 17px;
            font-weight: 600;
            color: #333;
            margin-bottom: 6px;
        }
        .product-category {
            font-size: 12px;
            color: #667eea;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .product-description {
            font-size: 13px;
            color: #888;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-price {
            font-size: 22px;
            font-weight: 700;
            color: #333;
        }
        .product-price span {
            font-size: 14px;
            color: #999;
            font-weight: 400;
        }
        .results-info {
            color: white;
            margin-bottom: 15px;
            font-size: 16px;
            opacity: 0.9;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: white;
        }
        .no-results h3 {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .no-results p {
            opacity: 0.8;
        }
        .back-link {
            color: white;
            text-decoration: none;
            display: inline-block;
            margin-top: 15px;
            padding: 10px 25px;
            border: 1px solid rgba(255,255,255,0.4);
            border-radius: 25px;
            transition: all 0.3s;
        }
        .back-link:hover {
            background: rgba(255,255,255,0.2);
        }
        .footer {
            text-align: center;
            padding: 30px;
            color: rgba(255,255,255,0.6);
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛍️ <span>Simple</span>Shop</h1>
        <div class="nav-links">
            <a href="/">Home</a>
            <a href="/products">All Products</a>
        </div>
    </div>

    <div class="hero">
        <h2>Find What You Need</h2>
        <p>Search our collection of products by name, category, or keyword</p>
        <div class="search-container">
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search for products..." value="{{ query or '' }}" autofocus>
                <button type="submit">Search</button>
            </form>
        </div>
        <div class="categories">
            <a href="/search?q=Electronics" class="category-btn">⚡ Electronics</a>
            <a href="/search?q=Accessories" class="category-btn">🎒 Accessories</a>
            <a href="/search?q=Home Office" class="category-btn">🏠 Home Office</a>
            <a href="/search?q=Stationery" class="category-btn">✏️ Stationery</a>
        </div>
    </div>

    <div class="container">
        {% if show_results %}
            {% if products %}
                <p class="results-info">Found {{ products|length }} result(s) for "{{ query }}"</p>
                <div class="products-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-icon">{{ product['image_url'] }}</div>
                        <div class="product-category">{{ product['category'] }}</div>
                        <div class="product-name">{{ product['name'] }}</div>
                        <div class="product-description">{{ product['description'] }}</div>
                        <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <h3>😕 No products found</h3>
                    <p>Try a different search term or browse our categories above.</p>
                    <a href="/" class="back-link">← Back to Home</a>
                </div>
            {% endif %}
        {% elif show_all %}
            <h3 class="section-title">All Products ({{ products|length }})</h3>
            <div class="products-grid">
                {% for product in products %}
                <div class="product-card">
                    <div class="product-icon">{{ product['image_url'] }}</div>
                    <div class="product-category">{{ product['category'] }}</div>
                    <div class="product-name">{{ product['name'] }}</div>
                    <div class="product-description">{{ product['description'] }}</div>
                    <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                </div>
                {% endfor %}
            </div>
        {% else %}
            <h3 class="section-title">Featured Products</h3>
            <div class="products-grid">
                {% for product in products %}
                <div class="product-card">
                    <div class="product-icon">{{ product['image_url'] }}</div>
                    <div class="product-category">{{ product['category'] }}</div>
                    <div class="product-name">{{ product['name'] }}</div>
                    <div class="product-description">{{ product['description'] }}</div>
                    <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>

    <div class="footer">
        <p>SimpleShop &copy; 2024 — A beginner-friendly Python web app</p>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products ORDER BY RANDOM() LIMIT 8')
    products = cursor.fetchall()
    conn.close()
    return render_template_string(HOME_TEMPLATE, products=products, query='', show_results=False, show_all=False)

@app.route('/products')
def all_products():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products ORDER BY category, name')
    products = cursor.fetchall()
    conn.close()
    return render_template_string(HOME_TEMPLATE, products=products, query='', show_results=False, show_all=True)

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    
    if not query:
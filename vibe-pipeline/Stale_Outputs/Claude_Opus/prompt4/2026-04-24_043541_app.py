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
            description TEXT NOT NULL,
            price REAL NOT NULL,
            image_url TEXT DEFAULT ''
        )
    ''')

    cursor.execute('SELECT COUNT(*) FROM products')
    count = cursor.fetchone()[0]

    if count == 0:
        sample_products = [
            ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 19.99, '🖱️'),
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 49.99, '⌨️'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 29.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better posture', 34.99, '💻'),
            ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with built-in microphone', 39.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness and color temperature', 24.99, '💡'),
            ('Noise Cancelling Headphones', 'Electronics', 'Over-ear headphones with active noise cancellation', 79.99, '🎧'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with non-slip rubber base', 12.99, '🖱️'),
            ('Phone Stand', 'Accessories', 'Adjustable phone stand for desk compatible with all phones', 9.99, '📱'),
            ('Bluetooth Speaker', 'Electronics', 'Portable Bluetooth speaker with 12-hour battery life', 29.99, '🔊'),
            ('Notebook Journal', 'Stationery', 'Premium leather-bound notebook with 200 lined pages', 14.99, '📓'),
            ('Gel Pen Set', 'Stationery', 'Set of 12 colorful gel pens for writing and drawing', 7.99, '🖊️'),
            ('Coffee Mug', 'Home Office', 'Ceramic coffee mug with funny programming quotes', 11.99, '☕'),
            ('Cable Organizer', 'Accessories', 'Silicone cable management clips for desk organization', 6.99, '📎'),
            ('Desk Organizer', 'Home Office', 'Wooden desk organizer with multiple compartments', 22.99, '🗂️'),
            ('Screen Cleaner Kit', 'Accessories', 'Screen cleaning spray with microfiber cloth', 8.99, '🧹'),
            ('Portable Charger', 'Electronics', '10000mAh portable power bank with fast charging', 24.99, '🔋'),
            ('Ergonomic Wrist Rest', 'Accessories', 'Memory foam wrist rest for keyboard comfort', 15.99, '🤲'),
            ('Book Light', 'Home Office', 'Rechargeable clip-on LED book reading light', 10.99, '📖'),
            ('Sticky Notes Pack', 'Stationery', 'Pack of 500 colorful sticky notes in 5 colors', 5.99, '📝'),
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
    <title>Mini Online Shop</title>
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
            color: #333;
        }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header-content {
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }
        .logo span {
            color: #764ba2;
        }
        nav a {
            text-decoration: none;
            color: #555;
            margin-left: 20px;
            font-weight: 500;
            transition: color 0.3s;
        }
        nav a:hover {
            color: #667eea;
        }
        .hero {
            text-align: center;
            padding: 60px 20px 40px;
            color: white;
        }
        .hero h1 {
            font-size: 42px;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .hero p {
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        .search-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 0 20px;
        }
        .search-form {
            display: flex;
            gap: 10px;
            background: white;
            padding: 8px;
            border-radius: 50px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .search-form input[type="text"] {
            flex: 1;
            border: none;
            padding: 14px 20px;
            font-size: 16px;
            outline: none;
            border-radius: 50px;
            background: transparent;
        }
        .search-form button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px 30px;
            font-size: 16px;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .search-form button:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.5);
        }
        .category-filter {
            text-align: center;
            margin-top: 20px;
        }
        .category-filter a {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            padding: 8px 18px;
            border-radius: 20px;
            margin: 5px;
            font-size: 14px;
            transition: background 0.3s;
        }
        .category-filter a:hover {
            background: rgba(255,255,255,0.4);
        }
        .container {
            max-width: 1000px;
            margin: 30px auto;
            padding: 0 20px 40px;
        }
        .results-info {
            color: white;
            text-align: center;
            margin-bottom: 20px;
            font-size: 16px;
        }
        .results-info span {
            font-weight: bold;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .product-icon {
            font-size: 48px;
            text-align: center;
            margin-bottom: 15px;
        }
        .product-category {
            display: inline-block;
            background: #f0f0ff;
            color: #667eea;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .product-name {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #222;
        }
        .product-description {
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .product-price {
            font-size: 24px;
            font-weight: 800;
            color: #667eea;
        }
        .product-price small {
            font-size: 14px;
            color: #999;
            font-weight: 400;
        }
        .add-to-cart-btn {
            display: block;
            width: 100%;
            margin-top: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.3s;
        }
        .add-to-cart-btn:hover {
            opacity: 0.9;
        }
        .no-results {
            text-align: center;
            color: white;
            padding: 60px 20px;
        }
        .no-results h2 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        .no-results p {
            font-size: 16px;
            opacity: 0.8;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: rgba(255,255,255,0.7);
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">🛍️ Mini<span>Shop</span></div>
            <nav>
                <a href="/">Home</a>
                <a href="/search?q=">All Products</a>
            </nav>
        </div>
    </div>

    <div class="hero">
        <h1>Welcome to MiniShop</h1>
        <p>Find the best products at amazing prices</p>
        <div class="search-container">
            <form class="search-form" action="/search" method="get">
                <input type="text" name="q" placeholder="Search products by name, category, or keyword..." value="{{ query or '' }}">
                <button type="submit">Search</button>
            </form>
        </div>
        <div class="category-filter">
            <a href="/search?q=Electronics">🔌 Electronics</a>
            <a href="/search?q=Accessories">🎒 Accessories</a>
            <a href="/search?q=Home Office">🏠 Home Office</a>
            <a href="/search?q=Stationery">📝 Stationery</a>
            <a href="/search?q=">📦 All Products</a>
        </div>
    </div>

    {% if products is not none %}
    <div class="container">
        {% if query %}
        <div class="results-info">
            Found <span>{{ products|length }}</span> result(s) for "<span>{{ query }}</span>"
        </div>
        {% else %}
        <div class="results-info">
            Showing <span>all {{ products|length }}</span> products
        </div>
        {% endif %}

        {% if products|length > 0 %}
        <div class="products-grid">
            {% for product in products %}
            <div class="product-card">
                <div class="product-icon">{{ product['image_url'] }}</div>
                <span class="product-category">{{ product['category'] }}</span>
                <div class="product-name">{{ product['name'] }}</div>
                <div class="product-description">{{ product['description'] }}</div>
                <div class="product-price">${{ "%.2f"|format(product['price']) }} <small>USD</small></div>
                <button class="add-to-cart-btn" onclick="alert('Added {{ product['name'] }} to cart!')">🛒 Add to Cart</button>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <h2>😕 No products found</h2>
            <p>Try searching with different keywords or browse our categories above.</p>
        </div>
        {% endif %}
    </div>
    {% endif %}

    <div class="footer">
        <p>&copy; 2024 MiniShop - A simple online shop demo</p>
    </div>
</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE, products=None, query=None)


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()

    conn = get_db()
    cursor = conn.cursor()

    if query:
        search_term = f'%{query}%'
        cursor.execute('''
            SELECT * FROM products
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
            ORDER BY name
        ''', (search_term, search_term, search_term))
    else:
        cursor.execute('SELECT * FROM products ORDER BY name')

    products = cursor.fetchall()
    conn.close()

    return render_template_string(HOME_TEMPLATE, products=products, query=query)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
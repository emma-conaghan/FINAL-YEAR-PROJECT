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
            ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 29.99, '🖱️'),
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 79.99, '⌨️'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 45.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better ergonomics', 39.99, '💻'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 59.99, '📷'),
            ('Noise Cancelling Headphones', 'Audio', 'Over-ear wireless headphones with active noise cancellation', 149.99, '🎧'),
            ('Bluetooth Speaker', 'Audio', 'Portable waterproof Bluetooth speaker', 34.99, '🔊'),
            ('Phone Case', 'Accessories', 'Shockproof silicone phone case for smartphones', 12.99, '📱'),
            ('Screen Protector', 'Accessories', 'Tempered glass screen protector 2-pack', 9.99, '🛡️'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness levels', 24.99, '💡'),
            ('Monitor Light Bar', 'Home Office', 'Screen light bar to reduce eye strain', 49.99, '🖥️'),
            ('Ergonomic Chair Cushion', 'Home Office', 'Memory foam seat cushion for office chairs', 29.99, '🪑'),
            ('Notebook Set', 'Stationery', 'Set of 3 lined notebooks with hard covers', 14.99, '📓'),
            ('Pen Set', 'Stationery', 'Premium ballpoint pen set with ink refills', 19.99, '🖊️'),
            ('Cable Organizer', 'Accessories', 'Silicone cable management clips 5-pack', 7.99, '🔗'),
            ('Power Bank', 'Electronics', '10000mAh portable charger with fast charging', 25.99, '🔋'),
            ('Mouse Pad XL', 'Accessories', 'Extended gaming mouse pad with non-slip base', 15.99, '🎮'),
            ('Wireless Earbuds', 'Audio', 'True wireless earbuds with charging case', 49.99, '🎵'),
            ('Coffee Mug', 'Home Office', 'Insulated stainless steel coffee mug 16oz', 18.99, '☕'),
            ('Desk Organizer', 'Home Office', 'Wooden desk organizer with multiple compartments', 22.99, '🗂️'),
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
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px 0;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        .header h1 {
            color: white;
            font-size: 2.5em;
            margin-bottom: 5px;
        }
        .header p {
            color: rgba(255, 255, 255, 0.8);
            font-size: 1.1em;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .search-box {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
            margin-bottom: 30px;
        }
        .search-box h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .search-form {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            flex: 1;
            min-width: 200px;
            padding: 14px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #667eea;
        }
        .search-form select {
            padding: 14px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            background: white;
            cursor: pointer;
            outline: none;
        }
        .search-form select:focus {
            border-color: #667eea;
        }
        .search-form button {
            padding: 14px 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .search-form button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .results-info {
            color: white;
            margin-bottom: 20px;
            font-size: 1.1em;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }
        .product-emoji {
            font-size: 3em;
            margin-bottom: 15px;
        }
        .product-name {
            font-size: 1.2em;
            color: #333;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .product-category {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 0.95em;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .product-price {
            font-size: 1.4em;
            color: #667eea;
            font-weight: 700;
        }
        .no-results {
            background: white;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .no-results h3 {
            color: #333;
            margin-bottom: 10px;
        }
        .no-results p {
            color: #666;
        }
        .browse-all {
            text-align: center;
            margin-top: 10px;
        }
        .browse-all a {
            color: rgba(255, 255, 255, 0.9);
            text-decoration: none;
            font-size: 1em;
            border-bottom: 1px solid rgba(255, 255, 255, 0.5);
            padding-bottom: 2px;
        }
        .browse-all a:hover {
            color: white;
            border-bottom-color: white;
        }
        .categories-bar {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .category-link {
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 0.9em;
            transition: background 0.3s, color 0.3s;
        }
        .category-link:hover {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛍️ Simple Online Shop</h1>
        <p>Find the best products at great prices</p>
    </div>

    <div class="container">
        <div class="search-box">
            <h2>🔍 Search Products</h2>
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search by name or keyword..." value="{{ query or '' }}">
                <select name="category">
                    <option value="">All Categories</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                    {% endfor %}
                </select>
                <button type="submit">Search</button>
            </form>
            <div class="categories-bar">
                <span style="color: #999; padding: 8px 0;">Quick browse:</span>
                {% for cat in categories %}
                <a href="/search?category={{ cat }}" class="category-link">{{ cat }}</a>
                {% endfor %}
            </div>
        </div>

        {% if products is not none %}
            {% if products|length > 0 %}
                <p class="results-info">
                    Found {{ products|length }} product{{ 's' if products|length != 1 else '' }}
                    {% if query %} for "{{ query }}"{% endif %}
                    {% if selected_category %} in {{ selected_category }}{% endif %}
                </p>
                <div class="products-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-emoji">{{ product['image_url'] }}</div>
                        <div class="product-name">{{ product['name'] }}</div>
                        <span class="product-category">{{ product['category'] }}</span>
                        <p class="product-description">{{ product['description'] }}</p>
                        <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <h3>😕 No products found</h3>
                    <p>Try a different search term or browse all categories.</p>
                </div>
            {% endif %}
            <div class="browse-all">
                <a href="/">← Back to Home</a>
            </div>
        {% else %}
            <p class="results-info">Browse our collection or search for something specific!</p>
            <div class="products-grid">
                {% for product in all_products %}
                <div class="product-card">
                    <div class="product-emoji">{{ product['image_url'] }}</div>
                    <div class="product-name">{{ product['name'] }}</div>
                    <span class="product-category">{{ product['category'] }}</span>
                    <p class="product-description">{{ product['description'] }}</p>
                    <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>
</body>
</html>
'''


def get_categories():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]
    conn.close()
    return categories


def get_all_products():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products ORDER BY name')
    products = cursor.fetchall()
    conn.close()
    return products


def search_products(query, category):
    conn = get_db()
    cursor = conn.cursor()

    conditions = []
    params = []

    if query:
        conditions.append('(name LIKE ? OR description LIKE ? OR category LIKE ?)')
        search_term = f'%{query}%'
        params.extend([search_term, search_term, search_term])

    if category:
        conditions.append('category = ?')
        params.append(category)

    if conditions:
        sql = 'SELECT * FROM products WHERE ' + ' AND '.join(conditions) + ' ORDER BY name'
    else:
        sql = 'SELECT * FROM products ORDER BY name'

    cursor.execute(sql, params)
    products = cursor.fetchall()
    conn.close()
    return products


@app.route('/')
def home():
    categories = get_categories()
    all_products = get_all_products()
    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=None,
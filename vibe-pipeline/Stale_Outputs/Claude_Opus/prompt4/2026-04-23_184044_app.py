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
            ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 19.99, ''),
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 49.99, ''),
            ('USB-C Hub', 'Electronics', '7-in-1 USB-C hub with HDMI and SD card reader', 29.99, ''),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better posture', 34.99, ''),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 39.99, ''),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness and color temperature', 24.99, ''),
            ('Notebook Set', 'Stationery', 'Set of 3 premium lined notebooks, A5 size', 12.99, ''),
            ('Ballpoint Pens', 'Stationery', 'Pack of 10 smooth writing ballpoint pens', 7.99, ''),
            ('Coffee Mug', 'Home Office', 'Large ceramic coffee mug, 16oz capacity', 9.99, ''),
            ('Monitor Riser', 'Accessories', 'Wooden monitor riser with storage compartments', 27.99, ''),
            ('Bluetooth Speaker', 'Electronics', 'Portable Bluetooth speaker with 10-hour battery life', 35.99, ''),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 14.99, ''),
            ('Phone Stand', 'Accessories', 'Adjustable phone stand for desk, compatible with all phones', 11.99, ''),
            ('Headphone Stand', 'Accessories', 'Sleek aluminum headphone stand', 18.99, ''),
            ('Sticky Notes', 'Stationery', 'Colorful sticky notes, 6 pads of 100 sheets each', 5.99, ''),
            ('Desk Organizer', 'Home Office', 'Bamboo desk organizer with multiple compartments', 22.99, ''),
            ('Cable Clips', 'Accessories', 'Set of 10 adhesive cable management clips', 6.99, ''),
            ('Wireless Charger', 'Electronics', 'Fast wireless charging pad compatible with Qi devices', 17.99, ''),
            ('Ergonomic Wrist Rest', 'Accessories', 'Memory foam wrist rest for keyboard', 13.99, ''),
            ('Whiteboard Markers', 'Stationery', 'Set of 8 dry erase whiteboard markers', 8.99, ''),
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
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 0;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        header h1 {
            font-size: 2em;
            margin-bottom: 5px;
        }
        header p {
            font-size: 1em;
            opacity: 0.9;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-section {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin: 30px auto;
            max-width: 700px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            text-align: center;
        }
        .search-section h2 {
            margin-bottom: 20px;
            color: #555;
            font-weight: 500;
        }
        .search-form {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            flex: 1;
            min-width: 250px;
            padding: 12px 20px;
            border: 2px solid #e1e5ee;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #667eea;
        }
        .search-form button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: opacity 0.3s, transform 0.2s;
        }
        .search-form button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        .categories {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .categories a {
            display: inline-block;
            padding: 6px 16px;
            background: #f0f0f5;
            color: #555;
            text-decoration: none;
            border-radius: 20px;
            font-size: 14px;
            transition: background 0.3s, color 0.3s;
        }
        .categories a:hover {
            background: #667eea;
            color: white;
        }
        .results-info {
            text-align: center;
            margin: 20px 0;
            color: #666;
            font-size: 16px;
        }
        .results-info strong {
            color: #667eea;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }
        .product-card h3 {
            color: #333;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        .product-card .category-badge {
            display: inline-block;
            padding: 3px 10px;
            background: #eef0ff;
            color: #667eea;
            border-radius: 12px;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .product-card .description {
            color: #777;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .product-card .price {
            font-size: 1.3em;
            font-weight: 700;
            color: #2d8f4e;
        }
        .product-card .add-to-cart {
            display: inline-block;
            margin-top: 12px;
            padding: 8px 20px;
            background: #2d8f4e;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
            transition: background 0.3s;
        }
        .product-card .add-to-cart:hover {
            background: #247a40;
        }
        .no-results {
            text-align: center;
            padding: 50px 20px;
            color: #999;
        }
        .no-results h3 {
            font-size: 1.3em;
            margin-bottom: 10px;
        }
        .back-link {
            display: inline-block;
            margin: 20px 0;
            color: #667eea;
            text-decoration: none;
            font-size: 15px;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #aaa;
            font-size: 14px;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ Simple Online Shop</h1>
        <p>Find the products you love</p>
    </header>

    <div class="container">
        <div class="search-section">
            <h2>Search for Products</h2>
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query if query else '' }}">
                <button type="submit">🔍 Search</button>
            </form>
            <div class="categories">
                <span style="color: #999; line-height: 32px;">Popular:</span>
                {% for cat in categories %}
                <a href="/search?q={{ cat }}">{{ cat }}</a>
                {% endfor %}
            </div>
        </div>

        {% if searched %}
            {% if query %}
                <a href="/" class="back-link">← Back to Home</a>
            {% endif %}

            {% if products|length > 0 %}
                <div class="results-info">
                    Found <strong>{{ products|length }}</strong> result{{ 's' if products|length != 1 else '' }}
                    {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
                </div>
                <div class="product-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <h3>{{ product['name'] }}</h3>
                        <span class="category-badge">{{ product['category'] }}</span>
                        <p class="description">{{ product['description'] }}</p>
                        <div class="price">${{ "%.2f"|format(product['price']) }}</div>
                        <button class="add-to-cart" onclick="alert('Added {{ product.name }} to cart!')">Add to Cart</button>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <h3>😔 No products found</h3>
                    <p>Try searching with different keywords</p>
                </div>
            {% endif %}
        {% else %}
            <div class="results-info">
                <strong>All Products</strong>
            </div>
            <div class="product-grid">
                {% for product in products %}
                <div class="product-card">
                    <h3>{{ product['name'] }}</h3>
                    <span class="category-badge">{{ product['category'] }}</span>
                    <p class="description">{{ product['description'] }}</p>
                    <div class="price">${{ "%.2f"|format(product['price']) }}</div>
                    <button class="add-to-cart" onclick="alert('Added {{ product.name }} to cart!')">Add to Cart</button>
                </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>

    <footer>
        <p>&copy; 2024 Simple Online Shop. Built with Flask & SQLite.</p>
    </footer>
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


def search_products(query):
    conn = get_db()
    cursor = conn.cursor()
    search_term = f'%{query}%'
    cursor.execute('''
        SELECT * FROM products 
        WHERE name LIKE ? 
        OR category LIKE ? 
        OR description LIKE ?
        ORDER BY name
    ''', (search_term, search_term, search_term))
    products = cursor.fetchall()
    conn.close()
    return products


def get_all_products():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products ORDER BY name')
    products = cursor.fetchall()
    conn.close()
    return products


@app.route('/')
def home():
    categories = get_categories()
    products = get_all_products()
    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=products,
        query='',
        searched=False
    )


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    categories = get_categories()

    if query:
        products = search_products(query)
    else:
        products = get_all_products()

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=products,
        query=query,
        searched=True
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
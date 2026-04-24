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
            ('Wireless Mouse', 'Electronics', 'A comfortable wireless mouse with ergonomic design', 29.99, 'mouse wireless computer accessory'),
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 79.99, 'keyboard mechanical gaming computer'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 45.99, 'usb hub adapter computer accessory'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 39.99, 'laptop stand desk accessory ergonomic'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 59.99, 'webcam camera video streaming'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 34.99, 'lamp light desk home office'),
            ('Notebook Set', 'Stationery', 'Pack of 3 premium lined notebooks', 12.99, 'notebook journal writing stationery'),
            ('Gel Pen Pack', 'Stationery', 'Set of 10 colorful gel pens', 8.99, 'pen gel writing stationery colorful'),
            ('Monitor Arm', 'Accessories', 'Single monitor arm with clamp mount', 49.99, 'monitor arm mount desk accessory'),
            ('Headphones', 'Electronics', 'Over-ear noise cancelling headphones', 99.99, 'headphones audio music noise cancelling'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 15.99, 'mousepad desk accessory gaming'),
            ('Phone Stand', 'Accessories', 'Adjustable phone and tablet stand', 19.99, 'phone tablet stand holder accessory'),
            ('Cable Organizer', 'Home Office', 'Desktop cable management box', 22.99, 'cable organizer desk tidy home office'),
            ('Sticky Notes', 'Stationery', 'Multi-color sticky notes 500 pack', 6.99, 'sticky notes reminder stationery'),
            ('Ergonomic Chair Cushion', 'Home Office', 'Memory foam seat cushion for office chairs', 44.99, 'cushion chair ergonomic comfort home office'),
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
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
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
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 30px;
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
            margin-top: 30px;
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
            border: 2px solid #667eea;
            color: #667eea;
            border-radius: 25px;
            text-decoration: none;
            transition: all 0.3s;
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
            box-shadow: 0 1px 5px rgba(0,0,0,0.05);
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
        }
        .product-card h3 {
            color: #333;
            margin-bottom: 8px;
        }
        .product-card .category-badge {
            display: inline-block;
            padding: 4px 12px;
            background: #e8eaff;
            color: #667eea;
            border-radius: 15px;
            font-size: 0.85em;
            margin-bottom: 10px;
        }
        .product-card .description {
            color: #666;
            font-size: 0.95em;
            margin-bottom: 12px;
            line-height: 1.5;
        }
        .product-card .price {
            font-size: 1.4em;
            font-weight: bold;
            color: #764ba2;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .no-results h3 {
            color: #999;
            margin-bottom: 10px;
        }
        .back-link {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 25px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 8px;
        }
        .back-link:hover {
            opacity: 0.9;
        }
        .all-products-link {
            margin-top: 20px;
            text-align: center;
        }
        .all-products-link a {
            color: #667eea;
            text-decoration: none;
            font-size: 1.1em;
        }
        .all-products-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛍️ Simple Online Shop</h1>
        <p>Find great products at amazing prices</p>
    </div>
    <div class="container">
        <div class="search-box">
            <h2>Search Products</h2>
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query or '' }}">
                <button type="submit">🔍 Search</button>
            </form>
        </div>

        <div class="categories">
            <h3>Browse by Category</h3>
            <div class="category-links">
                {% for cat in categories %}
                <a href="/search?q={{ cat }}">{{ cat }}</a>
                {% endfor %}
            </div>
        </div>

        <div class="all-products-link">
            <a href="/search?q=">View All Products</a>
        </div>

        {% if show_results %}
        <div class="results-info">
            {% if query %}
                <strong>{{ results|length }}</strong> result(s) found for "<em>{{ query }}</em>"
            {% else %}
                Showing all <strong>{{ results|length }}</strong> products
            {% endif %}
        </div>

        {% if results %}
        <div class="product-grid">
            {% for product in results %}
            <div class="product-card">
                <h3>{{ product['name'] }}</h3>
                <span class="category-badge">{{ product['category'] }}</span>
                <p class="description">{{ product['description'] }}</p>
                <div class="price">${{ "%.2f"|format(product['price']) }}</div>
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
        {% endif %}
    </div>
</body>
</html>
'''


@app.route('/')
def home():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]
    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        query='',
        results=[],
        show_results=False
    )


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]

    if query:
        search_term = f'%{query}%'
        cursor.execute('''
            SELECT * FROM products
            WHERE name LIKE ?
               OR category LIKE ?
               OR description LIKE ?
               OR keywords LIKE ?
            ORDER BY name
        ''', (search_term, search_term, search_term, search_term))
    else:
        cursor.execute('SELECT * FROM products ORDER BY name')

    results = cursor.fetchall()
    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        query=query,
        results=results,
        show_results=True
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
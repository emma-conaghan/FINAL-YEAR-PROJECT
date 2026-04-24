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
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 79.99, 'keyboard mechanical gaming typing'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 45.99, 'usb hub adapter dongle'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 34.99, 'laptop stand desk ergonomic'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 59.99, 'webcam camera video streaming'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 24.99, 'lamp light desk led'),
            ('Notebook Journal', 'Stationery', 'Hardcover lined notebook 200 pages', 12.99, 'notebook journal writing paper'),
            ('Gel Pen Set', 'Stationery', 'Set of 12 colored gel pens', 8.99, 'pen gel color writing'),
            ('Phone Stand', 'Accessories', 'Foldable phone stand for desk', 9.99, 'phone stand holder desk'),
            ('Bluetooth Speaker', 'Electronics', 'Portable Bluetooth speaker with bass boost', 39.99, 'speaker bluetooth portable music audio'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 14.99, 'mousepad mat desk gaming'),
            ('Screen Cleaner', 'Accessories', 'Screen cleaning kit with microfiber cloth', 7.99, 'screen cleaner cloth cleaning'),
            ('Coffee Mug', 'Home Office', 'Ceramic coffee mug 350ml - Programmer design', 11.99, 'mug coffee cup ceramic'),
            ('Cable Organizer', 'Accessories', 'Silicone cable management clips set of 5', 6.99, 'cable organizer clips management'),
            ('Whiteboard', 'Home Office', 'Magnetic dry erase whiteboard 60x40cm', 28.99, 'whiteboard magnetic dry erase office'),
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
            font-family: Arial, Helvetica, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        header h1 {
            margin-bottom: 5px;
            font-size: 28px;
        }
        header p {
            font-size: 14px;
            color: #bdc3c7;
        }
        .container {
            max-width: 900px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .search-box {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .search-box h2 {
            margin-bottom: 15px;
            color: #2c3e50;
        }
        .search-form {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .search-form input[type="text"] {
            flex: 1;
            min-width: 200px;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            outline: none;
        }
        .search-form input[type="text"]:focus {
            border-color: #3498db;
        }
        .search-form select {
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            outline: none;
            background: white;
        }
        .search-form button {
            padding: 12px 30px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .search-form button:hover {
            background-color: #2980b9;
        }
        .results-info {
            margin-bottom: 15px;
            color: #666;
            font-size: 14px;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        .product-card h3 {
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 18px;
        }
        .product-card .category {
            display: inline-block;
            background-color: #ecf0f1;
            color: #7f8c8d;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .product-card .description {
            color: #666;
            font-size: 14px;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-card .price {
            font-size: 22px;
            font-weight: bold;
            color: #27ae60;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        .no-results h3 {
            margin-bottom: 10px;
        }
        .browse-all {
            text-align: center;
            margin-top: 10px;
        }
        .browse-all a {
            color: #3498db;
            text-decoration: none;
        }
        .browse-all a:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 13px;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛒 Simple Online Shop</h1>
        <p>Find the products you need</p>
    </header>

    <div class="container">
        <div class="search-box">
            <h2>Search Products</h2>
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
            <div class="browse-all">
                <a href="/search?q=&category=">Browse all products</a>
            </div>
        </div>

        {% if products is not none %}
        <div class="results-info">
            {% if query or selected_category %}
                Found {{ products|length }} result(s)
                {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
                {% if selected_category %} in category "<strong>{{ selected_category }}</strong>"{% endif %}
            {% else %}
                Showing all {{ products|length }} product(s)
            {% endif %}
        </div>

        {% if products|length > 0 %}
        <div class="product-grid">
            {% for product in products %}
            <div class="product-card">
                <h3>{{ product['name'] }}</h3>
                <span class="category">{{ product['category'] }}</span>
                <p class="description">{{ product['description'] }}</p>
                <div class="price">${{ "%.2f"|format(product['price']) }}</div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-results">
            <h3>No products found</h3>
            <p>Try a different search term or category.</p>
        </div>
        {% endif %}
        {% endif %}
    </div>

    <footer>
        <p>&copy; 2024 Simple Online Shop - Built with Flask and SQLite</p>
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


@app.route('/')
def home():
    categories = get_categories()
    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=None,
        query=None,
        selected_category=None
    )


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    selected_category = request.args.get('category', '').strip()

    categories = get_categories()

    conn = get_db()
    cursor = conn.cursor()

    sql = 'SELECT * FROM products WHERE 1=1'
    params = []

    if query:
        sql += ' AND (name LIKE ? OR description LIKE ? OR keywords LIKE ?)'
        search_term = f'%{query}%'
        params.extend([search_term, search_term, search_term])

    if selected_category:
        sql += ' AND category = ?'
        params.append(selected_category)

    sql += ' ORDER BY name'

    cursor.execute(sql, params)
    products = cursor.fetchall()
    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=products,
        query=query,
        selected_category=selected_category
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
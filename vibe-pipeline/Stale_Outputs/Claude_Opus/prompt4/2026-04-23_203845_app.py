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
            ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 19.99, 'mouse wireless computer peripheral'),
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 49.99, 'keyboard mechanical gaming computer'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 29.99, 'usb hub adapter laptop accessory'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 34.99, 'laptop stand desk ergonomic'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 39.99, 'webcam camera video streaming'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 24.99, 'lamp light desk led office'),
            ('Notebook Journal', 'Stationery', 'Hardcover lined notebook 200 pages', 12.99, 'notebook journal writing paper'),
            ('Gel Pen Set', 'Stationery', 'Set of 12 colored gel pens', 8.99, 'pen gel color writing stationery'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 14.99, 'mousepad pad desk gaming'),
            ('Phone Holder', 'Accessories', 'Adjustable phone holder for desk', 9.99, 'phone holder stand mount desk'),
            ('Bluetooth Speaker', 'Electronics', 'Portable Bluetooth speaker with deep bass', 29.99, 'speaker bluetooth audio music portable'),
            ('Cable Organizer', 'Accessories', 'Silicone cable management clips set of 5', 6.99, 'cable organizer management desk tidy'),
            ('Monitor Arm', 'Accessories', 'Single monitor arm mount for desks', 44.99, 'monitor arm mount desk ergonomic display'),
            ('Whiteboard', 'Home Office', 'Magnetic dry erase whiteboard 24x36 inches', 27.99, 'whiteboard dry erase magnetic office'),
            ('Coffee Mug', 'Home Office', 'Insulated stainless steel coffee mug 16oz', 15.99, 'mug coffee cup insulated drink'),
            ('Headphone Stand', 'Accessories', 'Wooden headphone stand holder', 19.99, 'headphone stand holder wood desk'),
            ('Wireless Charger', 'Electronics', 'Fast wireless charging pad for smartphones', 17.99, 'charger wireless charging phone fast'),
            ('Sticky Notes', 'Stationery', 'Colorful sticky notes 3x3 inches 400 sheets', 5.99, 'sticky notes post reminder stationery'),
            ('Desk Organizer', 'Home Office', 'Wooden desk organizer with compartments', 22.99, 'desk organizer storage wood office'),
            ('Screen Cleaner', 'Accessories', 'Screen cleaning kit with microfiber cloth', 7.99, 'screen cleaner cloth spray display'),
        ]

        cursor.executemany('''
            INSERT INTO products (name, category, description, price, keywords)
            VALUES (?, ?, ?, ?, ?)
        ''', sample_products)

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
            font-family: Arial, sans-serif;
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
            margin-bottom: 20px;
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
            transition: border-color 0.3s;
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
            cursor: pointer;
        }
        .search-form button {
            padding: 12px 30px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
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
            grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
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
            background-color: #e8f4f8;
            color: #2980b9;
            padding: 3px 10px;
            border-radius: 12px;
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
        .categories-list {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .categories-list a {
            text-decoration: none;
            background: #ecf0f1;
            color: #2c3e50;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            transition: background 0.3s;
        }
        .categories-list a:hover {
            background: #3498db;
            color: white;
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
                <input type="text" name="q" placeholder="Search by name or keyword..." 
                       value="{{ query if query else '' }}">
                <select name="category">
                    <option value="">All Categories</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}" {{ 'selected' if selected_category == cat else '' }}>{{ cat }}</option>
                    {% endfor %}
                </select>
                <button type="submit">Search</button>
            </form>
            <div class="categories-list">
                <strong style="line-height: 35px;">Browse: </strong>
                {% for cat in categories %}
                <a href="/search?category={{ cat }}">{{ cat }}</a>
                {% endfor %}
            </div>
        </div>

        {% if searched %}
        <div class="results-info">
            {% if products %}
                Found {{ products|length }} product(s)
                {% if query %} matching "{{ query }}"{% endif %}
                {% if selected_category %} in {{ selected_category }}{% endif %}
            {% endif %}
        </div>
        {% endif %}

        {% if products %}
        <div class="product-grid">
            {% for product in products %}
            <div class="product-card">
                <h3>{{ product['name'] }}</h3>
                <span class="category">{{ product['category'] }}</span>
                <p class="description">{{ product['description'] }}</p>
                <p class="price">${{ "%.2f"|format(product['price']) }}</p>
            </div>
            {% endfor %}
        </div>
        {% elif searched %}
        <div class="no-results">
            <h3>No products found</h3>
            <p>Try a different search term or browse by category.</p>
        </div>
        {% else %}
        <h2 style="margin-bottom: 15px; color: #2c3e50;">All Products</h2>
        <div class="product-grid">
            {% for product in all_products %}
            <div class="product-card">
                <h3>{{ product['name'] }}</h3>
                <span class="category">{{ product['category'] }}</span>
                <p class="description">{{ product['description'] }}</p>
                <p class="price">${{ "%.2f"|format(product['price']) }}</p>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>

    <footer>
        <p>Simple Online Shop &copy; 2024 - Built with Flask and SQLite</p>
    </footer>
</body>
</html>
'''


@app.route('/')
def home():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]

    cursor.execute('SELECT * FROM products ORDER BY name')
    all_products = cursor.fetchall()

    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        all_products=all_products,
        products=None,
        searched=False,
        query='',
        selected_category=''
    )


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    category = request.args.get('category', '').strip()

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]

    sql = 'SELECT * FROM products WHERE 1=1'
    params = []

    if query:
        sql += ' AND (name LIKE ? OR description LIKE ? OR keywords LIKE ?)'
        search_term = f'%{query}%'
        params.extend([search_term, search_term, search_term])

    if category:
        sql += ' AND category = ?'
        params.append(category)

    sql += ' ORDER BY name'

    cursor.execute(sql, params)
    products = cursor.fetchall()

    conn.close()

    searched = bool(query or category)

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        all_products=[],
        products=products,
        searched=searched,
        query=query,
        selected_category=category
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
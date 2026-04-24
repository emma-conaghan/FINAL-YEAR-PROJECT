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
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI output', 49.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 39.99, '💻'),
            ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with microphone', 59.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 34.99, '💡'),
            ('Notebook Set', 'Stationery', 'Pack of 3 premium lined notebooks', 12.99, '📓'),
            ('Pen Set', 'Stationery', 'Set of 5 gel pens in assorted colors', 8.99, '🖊️'),
            ('Monitor Arm', 'Accessories', 'Single monitor desk mount arm', 44.99, '🖥️'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 15.99, '🎮'),
            ('Bluetooth Speaker', 'Electronics', 'Portable bluetooth speaker waterproof', 45.99, '🔊'),
            ('Phone Stand', 'Accessories', 'Adjustable phone holder for desk', 14.99, '📱'),
            ('Coffee Mug', 'Home Office', 'Large ceramic coffee mug programmer themed', 11.99, '☕'),
            ('Cable Organizer', 'Home Office', 'Desktop cable management clips set', 9.99, '📎'),
            ('Whiteboard', 'Home Office', 'Magnetic dry erase whiteboard 24x36 inches', 29.99, '📋'),
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
            background: #f0f2f5;
            color: #333;
            min-height: 100vh;
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
            padding: 12px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            width: 350px;
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
            margin: 20px auto;
            max-width: 700px;
            text-align: center;
        }
        .categories h3 {
            margin-bottom: 10px;
            color: #777;
            font-weight: 400;
            font-size: 0.9em;
        }
        .category-links {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .category-links a {
            display: inline-block;
            padding: 8px 18px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 20px;
            font-size: 14px;
            border: 1px solid #667eea;
            transition: all 0.3s;
        }
        .category-links a:hover {
            background: #667eea;
            color: white;
        }
        .results-info {
            margin: 20px 0 10px 0;
            color: #777;
            font-size: 0.95em;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }
        .product-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }
        .product-card h3 {
            margin-bottom: 8px;
            color: #333;
            font-size: 1.1em;
        }
        .product-category {
            display: inline-block;
            padding: 3px 10px;
            background: #f0f2f5;
            border-radius: 12px;
            font-size: 0.8em;
            color: #667eea;
            margin-bottom: 10px;
        }
        .product-description {
            color: #777;
            font-size: 0.9em;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        .product-price {
            font-size: 1.3em;
            font-weight: 700;
            color: #2d3436;
        }
        .no-results {
            text-align: center;
            padding: 50px;
            color: #999;
        }
        .no-results .emoji {
            font-size: 3em;
            margin-bottom: 15px;
        }
        .back-link {
            display: inline-block;
            margin-top: 15px;
            color: #667eea;
            text-decoration: none;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #aaa;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ Simple Online Shop</h1>
        <p>Find the products you need</p>
    </header>

    <div class="container">
        <div class="search-section">
            <h2>Search Products</h2>
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query or '' }}">
                <button type="submit">🔍 Search</button>
            </form>
        </div>

        <div class="categories">
            <h3>Browse by category:</h3>
            <div class="category-links">
                {% for cat in categories %}
                <a href="/search?q={{ cat }}">{{ cat }}</a>
                {% endfor %}
            </div>
        </div>

        {% if products is not none %}
            {% if products|length > 0 %}
                <p class="results-info">
                    Found <strong>{{ products|length }}</strong> result(s)
                    {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
                </p>
                <div class="product-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-icon">{{ product['image_url'] }}</div>
                        <h3>{{ product['name'] }}</h3>
                        <span class="product-category">{{ product['category'] }}</span>
                        <p class="product-description">{{ product['description'] }}</p>
                        <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <div class="emoji">😕</div>
                    <h3>No products found</h3>
                    <p>Try a different search term.</p>
                    <a href="/" class="back-link">← Back to home</a>
                </div>
            {% endif %}
        {% endif %}
    </div>

    <footer>
        <p>Simple Online Shop &copy; 2024 — Built with Flask & SQLite</p>
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
    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=None,
        query=''
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
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
            ORDER BY name
        ''', (search_term, search_term, search_term))
        products = cursor.fetchall()
    else:
        cursor.execute('SELECT * FROM products ORDER BY name')
        products = cursor.fetchall()

    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=products,
        query=query
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
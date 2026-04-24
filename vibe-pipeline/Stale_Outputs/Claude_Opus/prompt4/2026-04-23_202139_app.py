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
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical gaming keyboard with blue switches', 49.99, '⌨️'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and ethernet', 29.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better posture', 34.99, '💻'),
            ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with built-in microphone', 39.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness and color temperature', 24.99, '💡'),
            ('Notebook Journal', 'Stationery', 'Premium hardcover notebook with 200 lined pages', 12.99, '📓'),
            ('Gel Pen Set', 'Stationery', 'Set of 12 colorful gel pens for writing and drawing', 8.99, '🖊️'),
            ('Coffee Mug', 'Home Office', 'Ceramic coffee mug with funny programming quote', 14.99, '☕'),
            ('Mouse Pad', 'Accessories', 'Large extended mouse pad with stitched edges', 11.99, '🖥️'),
            ('Bluetooth Speaker', 'Electronics', 'Portable Bluetooth speaker with deep bass sound', 29.99, '🔊'),
            ('Phone Stand', 'Accessories', 'Adjustable phone stand holder for desk', 9.99, '📱'),
            ('Headphones', 'Electronics', 'Over-ear noise cancelling wireless headphones', 59.99, '🎧'),
            ('Desk Organizer', 'Home Office', 'Wooden desk organizer with multiple compartments', 22.99, '📦'),
            ('Sticky Notes', 'Stationery', 'Pack of 6 colorful sticky note pads', 5.99, '📝'),
            ('Water Bottle', 'Accessories', 'Insulated stainless steel water bottle 750ml', 18.99, '🍶'),
            ('Cable Clips', 'Accessories', 'Adhesive cable management clips pack of 10', 6.99, '📎'),
            ('Monitor Light Bar', 'Home Office', 'LED monitor light bar to reduce eye strain', 44.99, '🔆'),
            ('Wrist Rest', 'Accessories', 'Memory foam keyboard wrist rest pad', 13.99, '🤚'),
            ('Whiteboard', 'Home Office', 'Magnetic dry erase whiteboard 24x36 inches', 27.99, '📋'),
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
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-box {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin: 30px auto;
            max-width: 600px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
            border: 2px solid #667eea;
            border-radius: 20px;
            color: #667eea;
            text-decoration: none;
            transition: all 0.3s;
        }
        .category-links a:hover {
            background: #667eea;
            color: white;
        }
        .results-info {
            margin: 20px 0;
            color: #666;
            font-size: 1.1em;
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
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .product-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        .product-name {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 8px;
            color: #333;
        }
        .product-category {
            display: inline-block;
            padding: 3px 12px;
            background: #e8eaf6;
            color: #667eea;
            border-radius: 12px;
            font-size: 0.85em;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .product-price {
            font-size: 1.4em;
            font-weight: bold;
            color: #2e7d32;
        }
        .no-results {
            text-align: center;
            padding: 50px 20px;
            color: #999;
        }
        .no-results h3 {
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        .back-link {
            display: inline-block;
            margin: 20px 0;
            color: #667eea;
            text-decoration: none;
            font-size: 1.1em;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ Simple Online Shop</h1>
        <p>Find the products you need</p>
    </header>

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

        {% if show_results %}
            {% if query %}
                <a href="/" class="back-link">← Back to Home</a>
            {% endif %}

            {% if products %}
                <div class="results-info">
                    Found {{ products|length }} result(s){% if query %} for "<strong>{{ query }}</strong>"{% endif %}
                </div>
                <div class="product-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-icon">{{ product['image_url'] }}</div>
                        <div class="product-name">{{ product['name'] }}</div>
                        <span class="product-category">{{ product['category'] }}</span>
                        <div class="product-description">{{ product['description'] }}</div>
                        <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <h3>😕 No products found</h3>
                    <p>Try searching with different keywords</p>
                </div>
            {% endif %}
        {% endif %}
    </div>

    <footer>
        <p>&copy; 2024 Simple Online Shop - Built with Flask &amp; SQLite</p>
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
        products=[],
        query='',
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
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
            ORDER BY name
        ''', (search_term, search_term, search_term))
    else:
        cursor.execute('SELECT * FROM products ORDER BY name')

    products = cursor.fetchall()
    conn.close()

    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=products,
        query=query,
        show_results=True
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
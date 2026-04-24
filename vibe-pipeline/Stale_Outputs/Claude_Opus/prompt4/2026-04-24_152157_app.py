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
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C adapter with HDMI output', 29.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better posture', 34.99, '💻'),
            ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with built-in microphone', 39.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness levels', 24.99, '💡'),
            ('Notebook Set', 'Stationery', 'Pack of 3 premium lined notebooks for notes and journaling', 12.99, '📓'),
            ('Gel Pen Pack', 'Stationery', 'Set of 10 colorful gel pens for writing and drawing', 8.99, '🖊️'),
            ('Monitor Riser', 'Accessories', 'Wooden monitor riser with storage compartment', 27.99, '🖥️'),
            ('Bluetooth Speaker', 'Electronics', 'Portable waterproof bluetooth speaker with bass boost', 35.99, '🔊'),
            ('Phone Holder', 'Accessories', 'Universal adjustable phone holder for desk', 9.99, '📱'),
            ('Cable Organizer', 'Accessories', 'Silicone cable management clips set of 5', 6.99, '🔗'),
            ('Coffee Mug', 'Home Office', 'Large ceramic coffee mug with funny coding quote', 14.99, '☕'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 15.99, '🎯'),
            ('Headphone Stand', 'Accessories', 'Aluminum headphone stand with cable hook', 22.99, '🎧'),
            ('Whiteboard', 'Home Office', 'Small magnetic whiteboard with markers and eraser', 18.99, '📋'),
            ('Ergonomic Wrist Rest', 'Accessories', 'Memory foam wrist rest for keyboard comfort', 11.99, '🤲'),
            ('Plant Pot', 'Home Office', 'Minimalist ceramic plant pot for desk decoration', 16.99, '🪴'),
            ('Book Light', 'Home Office', 'Rechargeable clip-on reading light with warm LED', 13.99, '📖'),
            ('Sticky Notes', 'Stationery', 'Assorted color sticky notes 500 sheets pack', 5.99, '📝'),
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
        .search-box {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin: 30px auto;
            max-width: 700px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
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
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .search-form button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .categories {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .categories a {
            display: inline-block;
            padding: 6px 16px;
            background: #f0f2f5;
            color: #555;
            text-decoration: none;
            border-radius: 20px;
            font-size: 14px;
            transition: background 0.2s, color 0.2s;
        }
        .categories a:hover {
            background: #667eea;
            color: white;
        }
        .results-info {
            margin: 20px 0 10px 0;
            color: #666;
            font-size: 16px;
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
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        .product-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .product-name {
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }
        .product-category {
            display: inline-block;
            background: #e8ecff;
            color: #667eea;
            padding: 3px 12px;
            border-radius: 12px;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .product-description {
            color: #777;
            font-size: 14px;
            line-height: 1.5;
            margin-bottom: 15px;
        }
        .product-price {
            font-size: 22px;
            font-weight: 700;
            color: #667eea;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .no-results .emoji {
            font-size: 64px;
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
            font-size: 14px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🛍️ Simple Online Shop</h1>
        <p>Find the perfect products for your needs</p>
    </header>
    <div class="container">
        <div class="search-box">
            <h2>Search Products</h2>
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query or '' }}">
                <button type="submit">🔍 Search</button>
            </form>
            <div class="categories">
                <span style="color: #999; line-height: 32px;">Popular:</span>
                {% for cat in categories %}
                <a href="/search?q={{ cat }}">{{ cat }}</a>
                {% endfor %}
            </div>
        </div>

        {% if show_results %}
            {% if products %}
                <p class="results-info">Found <strong>{{ products|length }}</strong> result(s) for "<strong>{{ query }}</strong>"</p>
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
                    <div class="emoji">😕</div>
                    <h3>No products found</h3>
                    <p>Try a different search term or browse our categories above.</p>
                    <a href="/" class="back-link">← Back to Home</a>
                </div>
            {% endif %}
        {% else %}
            <p class="results-info">Browse all products:</p>
            <div class="product-grid">
                {% for product in all_products %}
                <div class="product-card">
                    <div class="product-icon">{{ product['image_url'] }}</div>
                    <div class="product-name">{{ product['name'] }}</div>
                    <span class="product-category">{{ product['category'] }}</span>
                    <div class="product-description">{{ product['description'] }}</div>
                    <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>
    <footer>
        <p>&copy; 2024 Simple Online Shop — Built with Flask &amp; SQLite</p>
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
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products ORDER BY name')
    all_products = cursor.fetchall()
    conn.close()

    categories = get_categories()

    return render_template_string(
        HOME_TEMPLATE,
        show_results=False,
        all_products=all_products,
        products=None,
        query='',
        categories=categories
    )


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()

    categories = get_categories()

    if not query:
        return render_template_string(
            HOME_TEMPLATE,
            show_results=False,
            all_products=[],
            products=[],
            query='',
            categories=categories
        )

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

    return render_template_string(
        HOME_TEMPLATE,
        show_results=True,
        all_products=[],
        products=products,
        query=query,
        categories=categories
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
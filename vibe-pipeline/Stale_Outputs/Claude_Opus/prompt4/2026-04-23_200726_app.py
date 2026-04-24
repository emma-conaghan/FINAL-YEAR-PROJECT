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
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 59.99, '⌨️'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 34.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand for better posture', 29.99, '💻'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 44.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness and color temperature', 24.99, '💡'),
            ('Notebook Set', 'Stationery', 'Pack of 3 premium lined notebooks for notes and journaling', 12.99, '📓'),
            ('Gel Pen Pack', 'Stationery', 'Set of 10 colored gel pens for writing and drawing', 8.99, '🖊️'),
            ('Monitor Riser', 'Accessories', 'Wooden monitor riser with storage compartments', 39.99, '🖥️'),
            ('Cable Organizer', 'Accessories', 'Silicone cable management clips for desk organization', 6.99, '🔗'),
            ('Bluetooth Speaker', 'Electronics', 'Portable Bluetooth speaker with 10-hour battery life', 29.99, '🔊'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 14.99, '🖱️'),
            ('Plant Pot', 'Home Office', 'Ceramic plant pot for desk decoration', 11.99, '🪴'),
            ('Coffee Mug', 'Home Office', 'Large ceramic coffee mug with funny coding quote', 9.99, '☕'),
            ('Headphone Stand', 'Accessories', 'Wooden headphone stand with cable holder', 22.99, '🎧'),
            ('Wireless Charger', 'Electronics', 'Fast wireless charging pad for smartphones', 18.99, '🔋'),
            ('Sticky Notes', 'Stationery', 'Colorful sticky notes pack of 500 sheets', 5.99, '📝'),
            ('Desk Organizer', 'Home Office', 'Bamboo desk organizer with multiple compartments', 27.99, '🗂️'),
            ('Screen Cleaner', 'Accessories', 'Screen cleaning kit with microfiber cloth and spray', 7.99, '✨'),
            ('Book Stand', 'Accessories', 'Adjustable book and tablet stand for reading', 16.99, '📖'),
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
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            padding: 40px 20px;
            color: white;
        }
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .search-box {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 30px;
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
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .results-info {
            color: white;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        .product-icon {
            font-size: 3em;
            margin-bottom: 12px;
        }
        .product-name {
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
            margin-bottom: 6px;
        }
        .product-category {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 0.9em;
            line-height: 1.5;
            margin-bottom: 12px;
        }
        .product-price {
            font-size: 1.4em;
            font-weight: 700;
            color: #667eea;
        }
        .no-results {
            text-align: center;
            color: white;
            padding: 40px;
            font-size: 1.2em;
        }
        .no-results span {
            font-size: 3em;
            display: block;
            margin-bottom: 15px;
        }
        .categories-bar {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 15px;
            justify-content: center;
        }
        .category-link {
            color: white;
            text-decoration: none;
            background: rgba(255,255,255,0.2);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            transition: background 0.3s;
        }
        .category-link:hover {
            background: rgba(255,255,255,0.4);
        }
        footer {
            text-align: center;
            color: rgba(255,255,255,0.7);
            padding: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛍️ Simple Online Shop</h1>
            <p>Search for products by name, category, or keyword</p>
            <div class="categories-bar">
                <a href="/" class="category-link">All Products</a>
                {% for cat in categories %}
                <a href="/search?category={{ cat }}" class="category-link">{{ cat }}</a>
                {% endfor %}
            </div>
        </header>

        <div class="search-box">
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="query" placeholder="Search for products..." value="{{ query or '' }}">
                <select name="category">
                    <option value="">All Categories</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                    {% endfor %}
                </select>
                <button type="submit">🔍 Search</button>
            </form>
        </div>

        {% if products is not none %}
            {% if products|length > 0 %}
                <div class="results-info">
                    Found {{ products|length }} product{{ 's' if products|length != 1 else '' }}
                    {% if query %} for "{{ query }}"{% endif %}
                    {% if selected_category %} in {{ selected_category }}{% endif %}
                </div>
                <div class="products-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-icon">{{ product['image_url'] }}</div>
                        <div class="product-name">{{ product['name'] }}</div>
                        <div class="product-category">{{ product['category'] }}</div>
                        <div class="product-description">{{ product['description'] }}</div>
                        <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <span>😕</span>
                    No products found
                    {% if query %} for "{{ query }}"{% endif %}
                    {% if selected_category %} in {{ selected_category }}{% endif %}
                    <br><br>
                    <a href="/" style="color: white;">← Browse all products</a>
                </div>
            {% endif %}
        {% endif %}

        <footer>
            Simple Online Shop &copy; 2024 — Built with Flask & SQLite
        </footer>
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


@app.route('/')
def home():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products ORDER BY name')
    products = cursor.fetchall()
    conn.close()
    categories = get_categories()
    return render_template_string(
        HOME_TEMPLATE,
        products=products,
        categories=categories,
        query='',
        selected_category=''
    )


@app.route('/search')
def search():
    query = request.args.get('query', '').strip()
    category = request.args.get('category', '').strip()

    conn = get_db()
    cursor = conn.cursor()

    sql = 'SELECT * FROM products WHERE 1=1'
    params = []

    if query:
        sql += ' AND (name LIKE ? OR description LIKE ? OR category LIKE ?)'
        search_term = f'%{query}%'
        params.extend([search_term, search_term, search_term])

    if category:
        sql += ' AND category = ?'
        params.append(category)

    sql += ' ORDER BY name'

    cursor.execute(sql, params)
    products = cursor.fetchall()
    conn.close()

    categories = get_categories()

    return render_template_string(
        HOME_TEMPLATE,
        products=products,
        categories=categories,
        query=query,
        selected_category=category
    )


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
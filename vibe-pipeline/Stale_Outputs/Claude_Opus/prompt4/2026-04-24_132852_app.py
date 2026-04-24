import sqlite3
from flask import Flask, request, render_template_string

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
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI output', 29.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 34.99, '💻'),
            ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with microphone', 39.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 24.99, '💡'),
            ('Notebook Set', 'Stationery', 'Pack of 3 lined notebooks for notes and journaling', 9.99, '📓'),
            ('Ballpoint Pens', 'Stationery', 'Set of 10 smooth writing ballpoint pens', 5.99, '🖊️'),
            ('Monitor Riser', 'Accessories', 'Wooden monitor riser with storage space', 29.99, '🖥️'),
            ('Headphones', 'Electronics', 'Over-ear noise cancelling headphones', 59.99, '🎧'),
            ('Mouse Pad XL', 'Accessories', 'Extra large mouse pad with stitched edges', 12.99, '🖱️'),
            ('Phone Charger', 'Electronics', 'Fast wireless phone charger pad', 15.99, '🔋'),
            ('Coffee Mug', 'Home Office', 'Ceramic coffee mug with funny programming quote', 11.99, '☕'),
            ('Cable Organizer', 'Accessories', 'Silicone cable management clips set of 5', 7.99, '🔗'),
            ('Ergonomic Chair Cushion', 'Home Office', 'Memory foam seat cushion for office chairs', 27.99, '🪑'),
            ('Blue Light Glasses', 'Accessories', 'Computer glasses that block blue light', 16.99, '👓'),
            ('Portable SSD', 'Electronics', '500GB portable solid state drive USB 3.0', 54.99, '💾'),
            ('Sticky Notes', 'Stationery', 'Colorful sticky notes pack of 6 pads', 4.99, '📝'),
            ('Desk Organizer', 'Home Office', 'Bamboo desk organizer with multiple compartments', 22.99, '🗂️'),
            ('Water Bottle', 'Accessories', 'Insulated stainless steel water bottle 750ml', 18.99, '🍶'),
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
            text-decoration: none;
            color: white;
        }
        .logo span {
            font-size: 32px;
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
            margin: 30px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
        }
        .search-section h2 {
            margin-bottom: 20px;
            color: #555;
            font-weight: 400;
            font-size: 20px;
        }
        .search-form {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .search-input {
            padding: 12px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            width: 400px;
            max-width: 100%;
            outline: none;
            transition: border-color 0.3s;
        }
        .search-input:focus {
            border-color: #667eea;
        }
        .search-button {
            padding: 12px 30px;
            font-size: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: opacity 0.3s;
            font-weight: 600;
        }
        .search-button:hover {
            opacity: 0.9;
        }
        .categories {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .category-link {
            display: inline-block;
            padding: 6px 16px;
            background: #f0f2f5;
            border-radius: 20px;
            text-decoration: none;
            color: #555;
            font-size: 14px;
            transition: all 0.3s;
        }
        .category-link:hover {
            background: #667eea;
            color: white;
        }
        .results-info {
            margin: 20px 0;
            color: #777;
            font-size: 15px;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        }
        .product-emoji {
            font-size: 48px;
            margin-bottom: 12px;
        }
        .product-name {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #333;
        }
        .product-category {
            display: inline-block;
            padding: 3px 10px;
            background: #eef0ff;
            color: #667eea;
            border-radius: 12px;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .product-description {
            color: #777;
            font-size: 14px;
            margin-bottom: 12px;
            line-height: 1.5;
        }
        .product-price {
            font-size: 22px;
            font-weight: 700;
            color: #2d3748;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .no-results .emoji {
            font-size: 64px;
            margin-bottom: 16px;
        }
        .no-results p {
            font-size: 18px;
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
        <div class="header-content">
            <a href="/" class="logo"><span>🛒</span> SimpleShop</a>
        </div>
    </header>

    <div class="container">
        <div class="search-section">
            <h2>Search our products by name, category, or keyword</h2>
            <form action="/search" method="GET" class="search-form">
                <input 
                    type="text" 
                    name="q" 
                    class="search-input" 
                    placeholder="Search for products..." 
                    value="{{ query or '' }}"
                    autofocus
                >
                <button type="submit" class="search-button">🔍 Search</button>
            </form>
            <div class="categories">
                <span style="color: #999; font-size: 14px; line-height: 32px;">Browse:</span>
                {% for cat in categories %}
                <a href="/search?q={{ cat }}" class="category-link">{{ cat }}</a>
                {% endfor %}
            </div>
        </div>

        {% if show_results %}
            {% if products %}
                <div class="results-info">
                    Found <strong>{{ products|length }}</strong> result(s) for "<strong>{{ query }}</strong>"
                </div>
                <div class="products-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-emoji">{{ product['image_url'] }}</div>
                        <div class="product-name">{{ product['name'] }}</div>
                        <div class="product-category">{{ product['category'] }}</div>
                        <div class="product-description">{{ product['description'] }}</div>
                        <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <div class="emoji">😕</div>
                    <p>No products found for "<strong>{{ query }}</strong>"</p>
                    <p style="font-size: 14px; margin-top: 10px;">Try a different search term or browse categories above.</p>
                </div>
            {% endif %}
        {% else %}
            <div class="results-info" style="text-align: center; margin-top: 30px;">
                <p>✨ Showing all products</p>
            </div>
            <div class="products-grid">
                {% for product in all_products %}
                <div class="product-card">
                    <div class="product-emoji">{{ product['image_url'] }}</div>
                    <div class="product-name">{{ product['name'] }}</div>
                    <div class="product-category">{{ product['category'] }}</div>
                    <div class="product-description">{{ product['description'] }}</div>
                    <div class="product-price">${{ "%.2f"|format(product['price']) }}</div>
                </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>

    <footer>
        &copy; 2024 SimpleShop — A beginner-friendly Python web app
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
        query='',
        show_results=False,
        products=[],
        all_products=all_products,
        categories=categories
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
            ORDER BY name
        ''', (search_term, search_term, search_term))
        products = cursor.fetchall()

        conn.close()

        return render_template_string(
            HOME_TEMPLATE,
            query=query,
            show_results=True,
            products=products,
            all_products=[],
            categories=categories
        )
    else:
        cursor.execute('SELECT * FROM products ORDER BY name')
        all_products = cursor.fetchall()
        conn.close()

        return render_template_string(
            HOME_TEMPLATE,
            query='',
            show_results=False,
            products=[],
            all_products=all_products,
            categories=categories
        )


if __name__ == '__main__':
    init_db()
    print("Shop database initialized!")
    print("Starting the Simple Online Shop...")
    print("Visit http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
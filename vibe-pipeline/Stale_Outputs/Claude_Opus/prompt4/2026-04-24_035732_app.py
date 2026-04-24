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
            ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 24.99, '🖱️'),
            ('Mechanical Keyboard', 'Electronics', 'RGB backlit mechanical gaming keyboard', 79.99, '⌨️'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C adapter with HDMI', 34.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 29.99, '💻'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with microphone', 49.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 19.99, '💡'),
            ('Notebook Journal', 'Stationery', 'Hardcover lined notebook 200 pages', 12.99, '📓'),
            ('Ballpoint Pen Set', 'Stationery', 'Set of 10 premium ballpoint pens', 8.99, '🖊️'),
            ('Coffee Mug', 'Home Office', 'Ceramic coffee mug with funny coding quote', 14.99, '☕'),
            ('Mouse Pad XL', 'Accessories', 'Extra large gaming mouse pad', 16.99, '🖥️'),
            ('Headphones', 'Electronics', 'Over-ear noise cancelling headphones', 89.99, '🎧'),
            ('Phone Stand', 'Accessories', 'Adjustable phone holder for desk', 11.99, '📱'),
            ('Sticky Notes', 'Stationery', 'Colorful sticky notes pack of 500', 6.99, '📝'),
            ('Desk Organizer', 'Home Office', 'Wooden desk organizer with compartments', 22.99, '🗄️'),
            ('Blue Light Glasses', 'Accessories', 'Computer glasses that block blue light', 18.99, '👓'),
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
            font-size: 1.1em;
            opacity: 0.9;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            margin: 30px auto;
            text-align: center;
        }
        .search-section h2 {
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
        .category-filter {
            margin-top: 15px;
        }
        .category-filter label {
            color: #777;
            margin-right: 8px;
        }
        .category-filter select {
            padding: 8px 15px;
            font-size: 14px;
            border: 2px solid #ddd;
            border-radius: 8px;
            outline: none;
            background: white;
        }
        .results-info {
            margin: 20px 0 10px 0;
            color: #777;
            font-size: 0.95em;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 25px rgba(0,0,0,0.12);
        }
        .product-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }
        .product-name {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .product-category {
            display: inline-block;
            background: #eef0ff;
            color: #667eea;
            padding: 3px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .product-description {
            color: #777;
            font-size: 0.9em;
            margin-bottom: 15px;
            line-height: 1.4;
        }
        .product-price {
            font-size: 1.4em;
            font-weight: bold;
            color: #2d9c4a;
        }
        .no-results {
            text-align: center;
            padding: 50px;
            color: #999;
        }
        .no-results span {
            font-size: 3em;
            display: block;
            margin-bottom: 15px;
        }
        .browse-all {
            text-align: center;
            margin-top: 15px;
        }
        .browse-all a {
            color: #667eea;
            text-decoration: none;
            font-size: 0.95em;
        }
        .browse-all a:hover {
            text-decoration: underline;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: #aaa;
            font-size: 0.9em;
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
                <input type="text" name="q" placeholder="Search by name, keyword..." value="{{ query or '' }}">
                <button type="submit">🔍 Search</button>
            </form>
            <div class="category-filter">
                <form action="/search" method="GET" style="display:inline;">
                    <label for="category">Or browse by category:</label>
                    <select name="category" id="category" onchange="this.form.submit()">
                        <option value="">All Categories</option>
                        {% for cat in categories %}
                        <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                        {% endfor %}
                    </select>
                </form>
            </div>
            <div class="browse-all">
                <a href="/search?q=">Browse all products</a>
            </div>
        </div>
        
        {% if products is not none %}
        <p class="results-info">
            {% if query %}
                Showing {{ products|length }} result(s) for "{{ query }}"
            {% elif selected_category %}
                Showing {{ products|length }} product(s) in "{{ selected_category }}"
            {% else %}
                Showing all {{ products|length }} product(s)
            {% endif %}
        </p>
        
        {% if products|length > 0 %}
        <div class="products-grid">
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
            <span>😕</span>
            <h3>No products found</h3>
            <p>Try a different search term or browse all products.</p>
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
        products=None,
        query=None,
        selected_category=None,
        categories=categories
    )

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    category = request.args.get('category', '').strip()
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]
    
    if category:
        cursor.execute(
            'SELECT * FROM products WHERE category = ? ORDER BY name',
            (category,)
        )
    elif query:
        search_term = '%' + query + '%'
        cursor.execute(
            '''SELECT * FROM products 
               WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
               ORDER BY name''',
            (search_term, search_term, search_term)
        )
    else:
        cursor.execute('SELECT * FROM products ORDER BY name')
    
    products = cursor.fetchall()
    conn.close()
    
    return render_template_string(
        HOME_TEMPLATE,
        products=products,
        query=query,
        selected_category=category,
        categories=categories
    )

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
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
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 79.99, 'keyboard mechanical gaming RGB computer'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI and USB 3.0', 45.99, 'USB hub adapter dongle computer'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 35.99, 'laptop stand holder desk ergonomic'),
            ('Webcam HD', 'Electronics', '1080p HD webcam with built-in microphone', 59.99, 'webcam camera video streaming'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 24.99, 'lamp LED light desk office'),
            ('Notebook Journal', 'Stationery', 'Hardcover lined notebook, 200 pages', 12.99, 'notebook journal writing paper'),
            ('Ballpoint Pen Set', 'Stationery', 'Set of 10 colored ballpoint pens', 8.99, 'pen writing stationery colors'),
            ('Monitor Stand', 'Accessories', 'Wooden monitor stand with storage', 49.99, 'monitor stand riser desk organizer'),
            ('Headphones', 'Electronics', 'Over-ear noise cancelling headphones', 99.99, 'headphones audio music noise cancelling'),
            ('Mouse Pad', 'Accessories', 'Large extended mouse pad for gaming', 15.99, 'mousepad gaming desk mat'),
            ('Phone Charger', 'Electronics', 'Fast wireless phone charger pad', 19.99, 'charger wireless phone fast charging'),
            ('Coffee Mug', 'Home Office', 'Insulated travel coffee mug, 16oz', 14.99, 'mug coffee travel insulated drink'),
            ('Desk Organizer', 'Home Office', 'Bamboo desk organizer with compartments', 22.99, 'organizer desk storage bamboo office'),
            ('Backpack', 'Accessories', 'Water-resistant laptop backpack', 54.99, 'backpack bag laptop travel waterproof'),
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
            font-size: 2em;
        }
        header p {
            font-size: 1em;
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
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 30px;
        }
        .search-box h2 {
            margin-bottom: 20px;
            color: #2c3e50;
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
            border-radius: 5px;
            width: 400px;
            max-width: 100%;
            outline: none;
            transition: border-color 0.3s;
        }
        .search-form input[type="text"]:focus {
            border-color: #3498db;
        }
        .search-form button {
            padding: 12px 30px;
            font-size: 16px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .search-form button:hover {
            background-color: #2980b9;
        }
        .categories {
            text-align: center;
            margin-top: 15px;
            font-size: 14px;
            color: #777;
        }
        .categories a {
            color: #3498db;
            text-decoration: none;
            margin: 0 8px;
        }
        .categories a:hover {
            text-decoration: underline;
        }
        .results-info {
            margin-bottom: 20px;
            font-size: 16px;
            color: #555;
        }
        .results-info strong {
            color: #2c3e50;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
            gap: 20px;
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .product-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        .product-card h3 {
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        .product-card .category-badge {
            display: inline-block;
            background-color: #e8f4fd;
            color: #3498db;
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
            font-size: 1.3em;
            font-weight: bold;
            color: #27ae60;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .no-results h3 {
            color: #e74c3c;
            margin-bottom: 10px;
        }
        .no-results p {
            color: #777;
        }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #3498db;
            text-decoration: none;
            font-size: 14px;
        }
        .back-link:hover {
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
                <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query or '' }}">
                <button type="submit">Search</button>
            </form>
            <div class="categories">
                Browse by category:
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
                Found <strong>{{ products|length }}</strong> result(s){% if query %} for "<strong>{{ query }}</strong>"{% endif %}
            </div>
            <div class="product-grid">
                {% for product in products %}
                <div class="product-card">
                    <h3>{{ product['name'] }}</h3>
                    <span class="category-badge">{{ product['category'] }}</span>
                    <p class="description">{{ product['description'] }}</p>
                    <p class="price">${{ "%.2f"|format(product['price']) }}</p>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="no-results">
                <h3>No products found</h3>
                <p>Try a different search term or browse by category above.</p>
            </div>
            {% endif %}
        {% endif %}
    </div>
    
    <footer>
        &copy; 2024 Simple Online Shop - Built with Flask &amp; SQLite
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
    products = cursor.fetchall()
    conn.close()
    
    return render_template_string(
        HOME_TEMPLATE,
        categories=categories,
        products=products,
        query='',
        show_results=True
    )

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row['category'] for row in cursor.fetchall()]
    
    if query:
        search_term = '%' + query + '%'
        cursor.execute('''
            SELECT * FROM products 
            WHERE name LIKE ? 
               OR category LIKE ? 
               OR description LIKE ? 
               OR keywords LIKE ?
            ORDER BY name
        ''', (search_term, search_term, search_term, search_term))
        products = cursor.fetchall()
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
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
            ('Wireless Mouse', 'Electronics', 'Ergonomic wireless mouse with USB receiver', 29.99, '🖱️'),
            ('Mechanical Keyboard', 'Electronics', 'RGB mechanical keyboard with blue switches', 79.99, '⌨️'),
            ('USB-C Hub', 'Electronics', 'Multi-port USB-C hub with HDMI output', 45.99, '🔌'),
            ('Laptop Stand', 'Accessories', 'Adjustable aluminum laptop stand', 35.99, '💻'),
            ('Webcam HD', 'Electronics', 'Full HD 1080p webcam with microphone', 59.99, '📷'),
            ('Desk Lamp', 'Home Office', 'LED desk lamp with adjustable brightness', 24.99, '💡'),
            ('Notebook Set', 'Stationery', 'Pack of 3 premium lined notebooks', 12.99, '📓'),
            ('Pen Set', 'Stationery', 'Professional ballpoint pen set of 5', 9.99, '🖊️'),
            ('Monitor Arm', 'Accessories', 'Single monitor arm mount clamp desk', 49.99, '🖥️'),
            ('Headphones', 'Electronics', 'Over-ear noise cancelling wireless headphones', 99.99, '🎧'),
            ('Mouse Pad XL', 'Accessories', 'Extra large gaming mouse pad', 15.99, '🎮'),
            ('Coffee Mug', 'Home Office', 'Insulated stainless steel coffee mug', 18.99, '☕'),
            ('Plant Pot', 'Home Office', 'Small ceramic desk plant pot', 11.99, '🪴'),
            ('Cable Organizer', 'Accessories', 'Silicone cable management clips set', 8.99, '🔗'),
            ('Bluetooth Speaker', 'Electronics', 'Portable mini bluetooth speaker waterproof', 39.99, '🔊'),
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
            outline: none;
            transition: border-color 0.3s;
        }
        .search-form input[type="text"]:focus {
            border-color: #667eea;
        }
        .search-form select {
            padding: 14px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            outline: none;
            background: white;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        .search-form select:focus {
            border-color: #667eea;
        }
        .search-form button {
            padding: 14px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .search-form button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .results-info {
            color: white;
            padding: 10px 20px;
            font-size: 1.1em;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
            padding: 10px 0;
        }
        .product-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .product-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        .product-name {
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }
        .product-category {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 15px;
            line-height: 1.4;
        }
        .product-price {
            font-size: 1.4em;
            font-weight: 700;
            color: #667eea;
        }
        .no-results {
            background: white;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .no-results h3 {
            color: #333;
            margin-bottom: 10px;
        }
        .no-results p {
            color: #666;
        }
        .categories-hint {
            color: rgba(255,255,255,0.8);
            text-align: center;
            margin-top: 10px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛍️ Simple Online Shop</h1>
            <p>Find the perfect products for your needs</p>
        </header>
        
        <div class="search-box">
            <form class="search-form" action="/search" method="GET">
                <input type="text" name="q" placeholder="Search by name or keyword..." value="{{ query or '' }}">
                <select name="category">
                    <option value="">All Categories</option>
                    {% for cat in categories %}
                    <option value="{{ cat }}" {% if selected_category == cat %}selected{% endif %}>{{ cat }}</option>
                    {% endfor %}
                </select>
                <button type="submit">🔍 Search</button>
            </form>
        </div>
        
        {% if searched %}
        <div class="results-info">
            Found <strong>{{ products|length }}</strong> result(s)
            {% if query %} for "<strong>{{ query }}</strong>"{% endif %}
            {% if selected_category %} in <strong>{{ selected_category }}</strong>{% endif %}
        </div>
        {% endif %}
        
        {% if products %}
        <div class="product-grid">
            {% for product in products %}
            <div class="product-card">
                <div class="product-icon">{{ product.image_url }}</div>
                <div class="product-name">{{ product.name }}</div>
                <span class="product-category">{{ product.category }}</span>
                <div class="product-description">{{ product.description }}</div>
                <div class="product-price">${{ "%.2f"|format(product.price) }}</div>
            </div>
            {% endfor %}
        </div>
        {% elif searched %}
        <div class="no-results">
            <h3>😔 No products found</h3>
            <p>Try a different search term or browse all categories.</p>
        </div>
        {% else %}
        <p class="categories-hint">Use the search box above to find products, or just click Search to browse all items!</p>
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
        products=[],
        categories=categories,
        query='',
        selected_category='',
        searched=False
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
    
    return render_template_string(
        HOME_TEMPLATE,
        products=products,
        categories=categories,
        query=query,
        selected_category=category,
        searched=True
    )

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
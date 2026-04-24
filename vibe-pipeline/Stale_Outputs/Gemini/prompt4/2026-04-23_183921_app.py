import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Initialize an in-memory database and populate it with sample products
def initialize_database():
    connection = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            keywords TEXT
        )
    ''')
    
    # Sample data for the shop
    shop_items = [
        (1, 'Gaming Laptop', 'Electronics', 'computer portable tech'),
        (2, 'Coffee Press', 'Kitchen', 'brew caffeine glass'),
        (3, 'Wireless Headphones', 'Electronics', 'audio music bluetooth'),
        (4, 'Yoga Mat', 'Fitness', 'exercise workout health'),
        (5, 'Ceramic Vase', 'Home Decor', 'flowers pottery interior'),
        (6, 'Running Shoes', 'Apparel', 'sports footwear sneakers')
    ]
    
    cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', shop_items)
    connection.commit()
    return connection

db = initialize_database()

# HTML template defined as a string for a single-file application
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: auto; padding: 20px; background-color: #f4f4f9; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .search-box { margin-bottom: 30px; }
        input[type="text"] { padding: 10px; width: 70%; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #218838; }
        .product { border-bottom: 1px solid #eee; padding: 15px 0; }
        .product:last-child { border-bottom: none; }
        .product-name { font-size: 1.2em; font-weight: bold; color: #007bff; }
        .product-meta { font-size: 0.9em; color: #666; }
        .no-results { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to Our Shop</h1>
        
        <div class="search-box">
            <form action="/" method="GET">
                <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query }}">
                <button type="submit">Search</button>
            </form>
        </div>

        <div class="results">
            {% if products %}
                <h2>Results:</h2>
                {% for p in products %}
                    <div class="product">
                        <div class="product-name">{{ p[1] }}</div>
                        <div class="product-meta">Category: {{ p[2] }} | Tags: {{ p[3] }}</div>
                    </div>
                {% endfor %}
            {% elif query %}
                <p class="no-results">No products found matching "{{ query }}".</p>
            {% else %}
                <p>Browse our catalog by entering a search term above.</p>
            {% endif %}
        </div>
        
        <p style="margin-top:20px;"><a href="/">View All Products</a></p>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    query = request.args.get('q', '').strip()
    cursor = db.cursor()
    
    if query:
        # Search for the term across name, category, or keywords
        search_param = f"%{query}%"
        cursor.execute('''
            SELECT * FROM products 
            WHERE name LIKE ? OR category LIKE ? OR keywords LIKE ?
        ''', (search_param, search_param, search_param))
        results = cursor.fetchall()
    else:
        # If no query, show everything or a subset
        cursor.execute('SELECT * FROM products')
        results = cursor.fetchall()
        
    return render_template_string(HTML_TEMPLATE, products=results, query=query)

if __name__ == '__main__':
    # Set to debug=True for development to see error logs
    app.run(host='0.0.0.0', port=5000, debug=True)
import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Initialize database and seed data
def init_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            keyword TEXT,
            price REAL
        )
    ''')
    products = [
        ('Smartphone X', 'Electronics', 'mobile tech phone', 699.99),
        ('Coffee Mug', 'Kitchen', 'ceramic drink coffee', 12.50),
        ('Wireless Headphones', 'Electronics', 'audio music bluetooth', 150.00),
        ('Yoga Mat', 'Fitness', 'exercise workout gym', 25.00),
        ('Mechanical Keyboard', 'Electronics', 'computer typing gaming', 89.99),
        ('Cooking Pan', 'Kitchen', 'chef frying nonstick', 45.00)
    ]
    cursor.executemany('INSERT INTO products (name, category, keyword, price) VALUES (?, ?, ?, ?)', products)
    conn.commit()
    return conn

db_conn = init_db()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Small Shop</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .product-card { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px; }
        .search-box { margin-bottom: 30px; }
        input[type="text"] { padding: 8px; width: 300px; }
        input[type="submit"] { padding: 8px 15px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Simple Online Shop</h1>
    
    <div class="search-box">
        <form action="/" method="GET">
            <input type="text" name="q" placeholder="Search for products, categories or keywords..." value="{{ query }}">
            <input type="submit" value="Search">
        </form>
    </div>

    <div class="results">
        {% if products %}
            <h2>Search Results</h2>
            {% for product in products %}
                <div class="product-card">
                    <h3>{{ product[1] }}</h3>
                    <p><strong>Category:</strong> {{ product[2] }}</p>
                    <p><strong>Price:</strong> ${{ "%.2f"|format(product[4]) }}</p>
                </div>
            {% endfor %}
        {% elif query %}
            <p>No products found for "{{ query }}".</p>
        {% else %}
            <p>Enter a term above to start shopping.</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    query = request.args.get('q', '')
    results = []
    
    if query:
        cursor = db_conn.cursor()
        # Search across name, category, and keyword using SQL LIKE
        sql_query = "SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR keyword LIKE ?"
        search_term = f"%{query}%"
        cursor.execute(sql_query, (search_term, search_term, search_term))
        results = cursor.fetchall()

    return render_template_string(HTML_TEMPLATE, products=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)
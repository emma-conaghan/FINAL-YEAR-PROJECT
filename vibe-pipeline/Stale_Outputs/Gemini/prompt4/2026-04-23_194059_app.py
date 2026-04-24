from flask import Flask, request, render_template_string
import sqlite3

app = Flask(__name__)

# Initialize an in-memory database for demonstration purposes
def init_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT,
            category TEXT,
            keyword TEXT
        )
    ''')
    products = [
        ('Ultra Laptop', 'Electronics', 'computer'),
        ('Wireless Mouse', 'Electronics', 'accessory'),
        ('Coffee Mug', 'Kitchen', 'drinkware'),
        ('Leather Notebook', 'Stationery', 'paper'),
        ('Cotton T-Shirt', 'Apparel', 'clothing'),
        ('Yoga Mat', 'Fitness', 'exercise')
    ]
    cursor.executemany('INSERT INTO products (name, category, keyword) VALUES (?, ?, ?)', products)
    conn.commit()
    return conn

db_connection = init_db()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .product-card { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 4px; }
        .search-box { margin-bottom: 30px; }
        input[type="text"] { padding: 8px; width: 300px; }
        input[type="submit"] { padding: 8px 15px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Shop Product Search</h1>
    
    <div class="search-box">
        <form method="GET" action="/">
            <input type="text" name="q" placeholder="Search by name, category, or keyword..." value="{{ query }}">
            <input type="submit" value="Search">
        </form>
    </div>

    {% if query %}
        <h2>Search Results for "{{ query }}"</h2>
        {% if products %}
            {% for product in products %}
            <div class="product-card">
                <strong>{{ product['name'] }}</strong><br>
                <small>Category: {{ product['category'] }} | Tag: {{ product['keyword'] }}</small>
            </div>
            {% endfor %}
        {% else %}
            <p>No products found matching your search.</p>
        {% endif %}
    {% else %}
        <p>Enter a search term above to find products.</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    query = request.args.get('q', '').strip()
    results = []
    
    if query:
        cursor = db_connection.cursor()
        # Simple search using LIKE across multiple columns
        search_term = f"%{query}%"
        cursor.execute(
            "SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR keyword LIKE ?", 
            (search_term, search_term, search_term)
        )
        results = cursor.fetchall()
        
    return render_template_string(HTML_TEMPLATE, products=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)
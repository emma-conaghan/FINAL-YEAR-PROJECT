import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Initialize an in-memory database for demonstration
db = sqlite3.connect(':memory:', check_same_thread=False)
cursor = db.cursor()
cursor.execute('CREATE TABLE products (id INTEGER, name TEXT, category TEXT, keyword TEXT)')

# Populate with some sample data
sample_products = [
    (1, 'Gaming Laptop', 'Electronics', 'computer'),
    (2, 'Wireless Mouse', 'Electronics', 'accessory'),
    (3, 'Ceramic Coffee Mug', 'Kitchen', 'cup'),
    (4, 'Cotton T-Shirt', 'Apparel', 'clothing'),
    (5, 'Stainless Steel Pan', 'Kitchen', 'cookware'),
    (6, 'Bluetooth Headphones', 'Electronics', 'audio')
]
cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', sample_products)
db.commit()

# Simple HTML template as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Mini Shop</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; line-height: 1.6; }
        .search-box { margin-bottom: 20px; padding: 20px; background: #f4f4f4; border-radius: 5px; }
        .product-item { border-bottom: 1px solid #ddd; padding: 10px 0; }
        input[type="text"] { padding: 8px; width: 70%; }
        input[type="submit"] { padding: 8px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Online Product Search</h1>
    
    <div class="search-box">
        <form action="/search" method="get">
            <input type="text" name="q" placeholder="Enter name, category, or keyword..." value="{{ query }}">
            <input type="submit" value="Search">
        </form>
    </div>

    {% if results is not none %}
        <h2>Search Results</h2>
        {% if results %}
            {% for item in results %}
                <div class="product-item">
                    <strong>{{ item[1] }}</strong><br>
                    <small>Category: {{ item[2] }} | Tags: {{ item[3] }}</small>
                </div>
            {% endfor %}
        {% else %}
            <p>No products found for "{{ query }}".</p>
        {% endif %}
        <p><a href="/">Clear search</a></p>
    {% else %}
        <p>Try searching for "Electronics", "Kitchen", or "Laptop".</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, results=None, query="")

@app.route('/search')
def search():
    user_query = request.args.get('q', '')
    
    # Use wildcards for a partial match
    search_param = f"%{user_query}%"
    
    cur = db.cursor()
    # Simple SQL query to check name, category, or keyword
    cur.execute(
        "SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR keyword LIKE ?",
        (search_param, search_param, search_param)
    )
    data = cur.fetchall()
    
    return render_template_string(HTML_TEMPLATE, results=data, query=user_query)

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Initialize the database and seed with data
def init_db():
    conn = sqlite3.connect('shop.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            keyword TEXT NOT NULL,
            price REAL NOT NULL
        )
    ''')
    # Clear existing data and seed for the demo
    cursor.execute('DELETE FROM products')
    products = [
        ('Wireless Mouse', 'Electronics', 'computer accessories laser clicker', 25.99),
        ('Yoga Mat', 'Fitness', 'exercise health workout rubber', 19.50),
        ('Coffee Beans', 'Grocery', 'organic beverage caffeine roast', 14.00),
        ('Desk Lamp', 'Home', 'office lighting furniture led', 45.00),
        ('Running Shoes', 'Apparel', 'sports footwear gym sneakers', 89.99),
        ('Hard Drive', 'Electronics', 'storage data memory hardware', 120.00)
    ]
    cursor.executemany('INSERT INTO products (name, category, keyword, price) VALUES (?, ?, ?, ?)', products)
    conn.commit()
    conn.close()

init_db()

# Simple HTML template as a string for single-file portability
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f4; }
        .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .search-box { margin-bottom: 30px; }
        input[type="text"] { padding: 10px; width: 300px; border: 1px solid #ddd; border-radius: 4px; }
        input[type="submit"] { padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .product-card { border-bottom: 1px solid #eee; padding: 15px 0; }
        .product-card:last-child { border-bottom: none; }
        .price { color: #b12704; font-weight: bold; }
        .category { color: #555; font-style: italic; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Mini Shop</h1>
        <div class="search-box">
            <form action="/search" method="get">
                <input type="text" name="q" placeholder="Search name, category, or keyword..." value="{{ query }}">
                <input type="submit" value="Search">
            </form>
        </div>

        <div id="results">
            {% if query %}
                <h3>Search results for "{{ query }}":</h3>
                {% if products %}
                    {% for p in products %}
                        <div class="product-card">
                            <strong>{{ p[1] }}</strong> <span class="category">({{ p[2] }})</span><br>
                            <span class="price">${{ "%.2f"|format(p[4]) }}</span>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No items found matching your search.</p>
                {% endif %}
            {% else %}
                <p>Welcome! Use the search box above to find products.</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, query="", products=[])

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    results = []
    if query:
        conn = sqlite3.connect('shop.db')
        cursor = conn.cursor()
        # Search using SQL LIKE across multiple columns
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT * FROM products 
            WHERE name LIKE ? 
            OR category LIKE ? 
            OR keyword LIKE ?
        """, (search_term, search_term, search_term))
        results = cursor.fetchall()
        conn.close()
    
    return render_template_string(HTML_PAGE, query=query, products=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Initialize an in-memory database for demonstration purposes
def init_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            keyword TEXT
        )
    ''')
    # Seed the shop with some sample data
    sample_products = [
        (1, 'Ultra Laptop', 'Electronics', 'computer portable technology high-end'),
        (2, 'Classic Coffee Mug', 'Kitchen', 'ceramic drinkware morning caffeine'),
        (3, 'Pro Running Shoes', 'Apparel', 'sports footwear gym exercise'),
        (4, 'Minimalist Desk Lamp', 'Furniture', 'office lighting decor study'),
        (5, 'Wireless Headphones', 'Electronics', 'audio music bluetooth gadget'),
        (6, 'Organic Green Tea', 'Groceries', 'healthy drink herbal antioxidant')
    ]
    cursor.executemany('INSERT INTO products VALUES (?,?,?,?)', sample_products)
    conn.commit()
    return conn

db = init_db()

# Simple HTML template as a string for single-file structure
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }
        .search-container { background: #f8f9fa; padding: 30px; border-radius: 10px; border: 1px solid #dee2e6; margin-bottom: 30px; }
        input[type="text"] { width: 70%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        .product-card { border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 5px; transition: shadow 0.3s; }
        .product-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .product-name { font-size: 1.2em; font-weight: bold; color: #007bff; }
        .product-meta { color: #6c757d; font-size: 0.9em; margin-top: 5px; }
        .no-results { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <h1>My Simple Shop</h1>
    
    <div class="search-container">
        <form method="GET">
            <input type="text" name="query" placeholder="Search by name, category, or keyword..." value="{{ user_query }}">
            <button type="submit">Search</button>
        </form>
    </div>

    {% if results %}
        <h2>Showing results for: "{{ user_query }}"</h2>
        {% for item in results %}
            <div class="product-card">
                <div class="product-name">{{ item[1] }}</div>
                <div class="product-meta">
                    <strong>Category:</strong> {{ item[2] }} | 
                    <strong>Tags:</strong> {{ item[3] }}
                </div>
            </div>
        {% endfor %}
    {% elif user_query %}
        <p class="no-results">No products found matching "{{ user_query }}".</p>
    {% else %}
        <p>Welcome! Use the search box above to browse our inventory.</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    user_query = request.args.get('query', '').strip()
    results = []
    
    if user_query:
        cursor = db.cursor()
        # Search across name, category, and keyword columns
        # Note: Using parameterized queries to prevent SQL injection
        sql_pattern = f"%{user_query}%"
        cursor.execute(
            "SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR keyword LIKE ?",
            (sql_pattern, sql_pattern, sql_pattern)
        )
        results = cursor.fetchall()
    
    return render_template_string(
        HTML_TEMPLATE, 
        results=results, 
        user_query=user_query
    )

if __name__ == '__main__':
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
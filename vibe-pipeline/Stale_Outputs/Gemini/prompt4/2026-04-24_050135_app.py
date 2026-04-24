import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Function to setup and populate an in-memory database
def initialize_database():
    connection = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            description TEXT
        )
    ''')
    
    # Adding sample shop items
    products = [
        (1, 'Coffee Maker', 'Appliances', 'Automatic drip coffee maker with glass carafe.'),
        (2, 'Denim Jacket', 'Clothing', 'Classic blue denim jacket with metal buttons.'),
        (3, 'Wireless Headphones', 'Electronics', 'Over-ear noise cancelling wireless headphones.'),
        (4, 'Yoga Mat', 'Fitness', 'Non-slip eco-friendly yoga mat for exercise.'),
        (5, 'Cast Iron Skillet', 'Kitchen', 'Pre-seasoned 12-inch heavy duty cast iron skillet.'),
        (6, 'Office Chair', 'Furniture', 'Ergonomic mesh office chair with lumbar support.')
    ]
    cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', products)
    connection.commit()
    return connection

# Initialize database connection globally for simplicity
db_conn = initialize_database()

# Combined HTML layout for search and results
PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }
        .header { text-align: center; margin-bottom: 40px; }
        .search-container { background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .search-form { display: flex; gap: 10px; }
        input[type="text"] { flex-grow: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #218838; }
        .product-card { border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .product-name { color: #007bff; margin: 0; }
        .product-meta { color: #666; font-size: 0.85em; margin: 5px 0; font-weight: bold; }
        .no-results { text-align: center; color: #999; margin-top: 50px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Welcome to MiniShop</h1>
        <p>Find the best products at the lowest prices.</p>
    </div>

    <div class="search-container">
        <form action="/search" method="get" class="search-form">
            <input type="text" name="keyword" placeholder="Search by name, category, or keyword..." value="{{ user_query }}">
            <button type="submit">Search</button>
        </form>
    </div>

    {% if results is not none %}
        <h3>Results for: "{{ user_query }}"</h3>
        {% if results %}
            {% for row in results %}
            <div class="product-card">
                <h2 class="product-name">{{ row[1] }}</h2>
                <p class="product-meta">Category: {{ row[2] }}</p>
                <p class="product-desc">{{ row[3] }}</p>
            </div>
            {% endfor %}
        {% else %}
            <div class="no-results">
                <p>No products found matching your criteria.</p>
            </div>
        {% endif %}
        <p><a href="/">Clear search and return home</a></p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    # Show the main search page without results
    return render_template_string(PAGE_HTML, results=None, user_query="")

@app.route('/search')
def search_products():
    # Extract search term from the URL query parameters
    user_input = request.args.get('keyword', '')
    
    # Prepare database cursor
    cur = db_conn.cursor()
    
    # Use SQL LIKE to search across name, category, and description
    # Percentage signs allow for partial matches
    search_term = f"%{user_input}%"
    query_string = """
        SELECT * FROM products 
        WHERE name LIKE ? 
        OR category LIKE ? 
        OR description LIKE ?
    """
    
    cur.execute(query_string, (search_term, search_term, search_term))
    items = cur.fetchall()
    
    return render_template_string(PAGE_HTML, results=items, user_query=user_input)

if __name__ == '__main__':
    # Start the Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)
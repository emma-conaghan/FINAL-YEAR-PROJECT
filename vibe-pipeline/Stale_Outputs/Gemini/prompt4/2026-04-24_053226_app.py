import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Initialize an in-memory database and populate it with sample data
def init_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE products (name TEXT, category TEXT, keyword TEXT)')
    items = [
        ('Coffee Grinder', 'Kitchen', 'electric, beans, brew'),
        ('Leather Notebook', 'Stationery', 'writing, office, paper'),
        ('Wireless Mouse', 'Electronics', 'computer, pc, gadget'),
        ('Running Sneakers', 'Footwear', 'sport, gym, fitness'),
        ('Ceramic Vase', 'Home Decor', 'flower, pottery, interior')
    ]
    cursor.executemany('INSERT INTO products VALUES (?, ?, ?)', items)
    conn.commit()
    return conn

db = init_db()

# HTML template combining search box and results display
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Online Shop</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; background-color: #f4f4f4; }
        .container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        input[type="text"] { width: 300px; padding: 10px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; }
        input[type="submit"] { padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .product-card { border: 1px solid #eee; padding: 10px; margin-top: 10px; border-radius: 5px; background: #fafafa; }
        .tag { font-size: 0.8em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Product Search</h1>
        <form action="/search" method="get">
            <input type="text" name="query" placeholder="Enter name, category or keyword..." required>
            <input type="submit" value="Search">
        </form>

        {% if searched %}
            <hr>
            <h2>Results for "{{ query_text }}"</h2>
            {% if items %}
                {% for item in items %}
                    <div class="product-card">
                        <strong>{{ item['name'] }}</strong><br>
                        Category: {{ item['category'] }}<br>
                        <span class="tag">Keywords: {{ item['keyword'] }}</span>
                    </div>
                {% endfor %}
            {% else %}
                <p>No products found matching your search.</p>
            {% endif %}
            <p><a href="/">Back to search</a></p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    # Render the initial page with only the search box
    return render_template_string(HTML_PAGE, searched=False)

@app.route('/search')
def search():
    # Handle the search request and fetch results from SQLite
    query_text = request.args.get('query', '')
    cursor = db.cursor()
    
    # Simple SQL search using LIKE for name, category, or keyword
    sql = "SELECT * FROM products WHERE name LIKE ? OR category LIKE ? OR keyword LIKE ?"
    wildcard_query = f"%{query_text}%"
    cursor.execute(sql, (wildcard_query, wildcard_query, wildcard_query))
    
    results = cursor.fetchall()
    return render_template_string(
        HTML_PAGE, 
        items=results, 
        searched=True, 
        query_text=query_text
    )

if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=5000)
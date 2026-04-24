from flask import Flask, render_template_string, request
import sqlite3

app = Flask(__name__)

DATABASE = 'shop.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            price REAL
        )
    ''')
    # Basic sample products
    c.execute("SELECT COUNT(*) FROM products")
    if c.fetchone()[0] == 0:
        products = [
            ('T-Shirt', 'Clothing', 'Soft cotton T-shirt', 12.99),
            ('Coffee Mug', 'Accessories', 'Ceramic mug', 8.50),
            ('Notebook', 'Stationery', 'Lined notebook', 5.99),
            ('Sneakers', 'Footwear', 'Comfortable sneakers', 49.99),
            ('Backpack', 'Bags', 'Durable backpack', 34.90),
            ('Pen', 'Stationery', 'Smooth blue pen', 1.20),
        ]
        for p in products:
            c.execute("INSERT INTO products (name, category, description, price) VALUES (?, ?, ?, ?)", p)
    conn.commit()
    conn.close()

@app.route('/', methods=['GET'])
def home():
    template = '''
    <html>
        <head>
            <title>Simple Online Shop</title>
        </head>
        <body>
            <h1>Welcome to the Online Shop</h1>
            <form action="/search" method="get">
                <input type="text" name="query" placeholder="Search products..." required>
                <input type="submit" value="Search">
            </form>
        </body>
    </html>
    '''
    return render_template_string(template)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    results = []
    if query:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        q = f"%{query}%"
        c.execute("""
            SELECT name, category, description, price FROM products
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        """, (q, q, q))
        results = c.fetchall()
        conn.close()
    template = '''
    <html>
        <head>
            <title>Search Results</title>
        </head>
        <body>
            <h1>Search Results</h1>
            <form action="/search" method="get">
                <input type="text" name="query" value="{{ query }}" placeholder="Search products..." required>
                <input type="submit" value="Search">
            </form>
            {% if results %}
                <ul>
                {% for r in results %}
                    <li>
                        <strong>{{ r[0] }}</strong> - {{ r[1] }}<br>
                        {{ r[2] }}<br>
                        Price: ${{ '%.2f' % r[3] }}
                    </li>
                {% endfor %}
                </ul>
            {% else %}
                <p>No matching products found.</p>
            {% endif %}
            <a href="/">Back to Home</a>
        </body>
    </html>
    '''
    return render_template_string(template, results=results, query=query)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
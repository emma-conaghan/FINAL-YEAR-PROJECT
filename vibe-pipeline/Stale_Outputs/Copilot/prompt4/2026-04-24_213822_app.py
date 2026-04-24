from flask import Flask, render_template_string, request
import sqlite3

app = Flask(__name__)

HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>Online Shop Search</title>
</head>
<body>
    <h1>Search Products</h1>
    <form method="GET" action="/search">
        <input type="text" name="q" placeholder="Product name, category or keyword" required>
        <input type="submit" value="Search">
    </form>
</body>
</html>
'''

HTML_RESULTS = '''
<!DOCTYPE html>
<html>
<head>
    <title>Search Results</title>
</head>
<body>
    <h1>Results for "{{ query }}"</h1>
    {% if results %}
        <ul>
        {% for product in results %}
            <li>
                <strong>{{ product[1] }}</strong> - {{ product[2] }}<br>
                Category: {{ product[3] }}<br>
                Description: {{ product[4] }}
            </li>
        {% endfor %}
        </ul>
    {% else %}
        <p>No products found.</p>
    {% endif %}
    <a href="/">Back to search</a>
</body>
</html>
'''

def init_db():
    conn = sqlite3.connect('shop.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL,
            category TEXT,
            description TEXT
        )
    ''')
    # Sample data
    c.execute("SELECT COUNT(*) FROM products")
    if c.fetchone()[0] == 0:
        products = [
            ("Red Shirt", 19.99, "Clothing", "A bright red shirt"),
            ("Blue Jeans", 29.99, "Clothing", "Comfortable blue jeans"),
            ("Wireless Mouse", 14.99, "Electronics", "Mouse with no wires"),
            ("Coffee Mug", 9.99, "Kitchen", "Mug for coffee, 350ml"),
            ("Running Shoes", 49.99, "Footwear", "Shoes for running"),
            ("Laptop Sleeve", 15.99, "Electronics", "Protective sleeve for laptops"),
            ("Green Jacket", 39.99, "Clothing", "Warm green jacket"),
            ("Toaster", 24.99, "Kitchen", "2-slice toaster"),
            ("Black Socks", 5.99, "Footwear", "Soft black socks"),
            ("Bluetooth Speaker", 34.99, "Electronics", "Portable speaker")
        ]
        for p in products:
            c.execute("INSERT INTO products (name, price, category, description) VALUES (?, ?, ?, ?)", p)
    conn.commit()
    conn.close()

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_FORM)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    results = []
    if query:
        conn = sqlite3.connect('shop.db')
        c = conn.cursor()
        sql = '''
            SELECT * FROM products
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ?
        '''
        like_query = f"%{query}%"
        c.execute(sql, (like_query, like_query, like_query))
        results = c.fetchall()
        conn.close()
    return render_template_string(HTML_RESULTS, query=query, results=results)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
from flask import Flask, request, render_template_string
import sqlite3

app = Flask(__name__)

DB_NAME = 'shop.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    category TEXT,
                    description TEXT
                )''')
    c.execute("INSERT INTO products (name, category, description) VALUES ('Laptop', 'Electronics', 'Portable computer')")
    c.execute("INSERT INTO products (name, category, description) VALUES ('Headphones', 'Electronics', 'Noise cancelling')")
    c.execute("INSERT INTO products (name, category, description) VALUES ('Coffee Mug', 'Home', 'Ceramic mug')")
    c.execute("INSERT INTO products (name, category, description) VALUES ('T-shirt', 'Clothing', 'Cotton shirt')")
    c.execute("INSERT INTO products (name, category, description) VALUES ('Desk Chair', 'Furniture', 'Comfortable chair')")
    conn.commit()
    conn.close()

@app.before_first_request
def setup():
    init_db()

SEARCH_HTML = """
<!doctype html>
<html>
  <head>
    <title>Shop Search</title>
  </head>
  <body>
    <h1>Online Shop Search</h1>
    <form action="/results" method="get">
      <input type="text" name="q" placeholder="Search products...">
      <button type="submit">Search</button>
    </form>
  </body>
</html>
"""

RESULTS_HTML = """
<!doctype html>
<html>
  <head>
    <title>Search Results</title>
  </head>
  <body>
    <h1>Results for "{{query}}"</h1>
    {% if results %}
      <ul>
      {% for row in results %}
        <li><strong>{{row[1]}}</strong> ({{row[2]}}) - {{row[3]}}</li>
      {% endfor %}
      </ul>
    {% else %}
      <p>No products found.</p>
    {% endif %}
    <a href="/">Back to search</a>
  </body>
</html>
"""

@app.route('/')
def search():
    return render_template_string(SEARCH_HTML)

@app.route('/results')
def results():
    query = request.args.get('q', '').strip()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    sql = '''SELECT * FROM products WHERE
             name LIKE ? OR
             category LIKE ? OR
             description LIKE ?'''
    params = ['%'+query+'%']*3
    c.execute(sql, params)
    rows = c.fetchall()
    conn.close()
    return render_template_string(RESULTS_HTML, results=rows, query=query)

if __name__ == '__main__':
    app.run(debug=True)
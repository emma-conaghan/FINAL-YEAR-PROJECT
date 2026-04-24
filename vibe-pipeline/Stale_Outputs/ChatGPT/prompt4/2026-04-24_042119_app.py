from flask import Flask, request, render_template_string

app = Flask(__name__)

products = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": "red shirt cotton"},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": "blue jeans denim"},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": "mug coffee cup"},
    {"id": 4, "name": "Wireless Mouse", "category": "Electronics", "keywords": "mouse wireless computer"},
    {"id": 5, "name": "HD Monitor", "category": "Electronics", "keywords": "monitor hd screen"},
]

search_page = """
<!doctype html>
<title>Simple Shop Search</title>
<h1>Search for Products</h1>
<form action="/results" method="get">
  <input type="text" name="query" placeholder="Enter name, category or keywords" required>
  <input type="submit" value="Search">
</form>
"""

results_page = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
{% if results %}
  <ul>
  {% for product in results %}
    <li><strong>{{ product.name }}</strong> - Category: {{ product.category }}</li>
  {% endfor %}
  </ul>
{% else %}
  <p>No products found.</p>
{% endif %}
<a href="/">New Search</a>
"""

@app.route('/')
def search():
    return render_template_string(search_page)

@app.route('/results')
def results():
    query = request.args.get('query', '').strip().lower()
    filtered = []
    if query:
        for product in products:
            if query in product["name"].lower() or query in product["category"].lower() or query in product["keywords"].lower():
                filtered.append(product)
    return render_template_string(results_page, results=filtered)

if __name__ == '__main__':
    app.run(debug=True)
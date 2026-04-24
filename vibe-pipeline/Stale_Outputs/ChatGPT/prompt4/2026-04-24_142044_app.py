from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"id": 1, "name": "Red T-shirt", "category": "Clothing", "keywords": ["red", "shirt", "tee"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "denim", "pants"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["coffee", "mug", "cup"]},
    {"id": 4, "name": "Gaming Mouse", "category": "Electronics", "keywords": ["gaming", "mouse", "computer"]},
    {"id": 5, "name": "Book: Python Basics", "category": "Books", "keywords": ["python", "programming", "book"]},
]

search_page = """
<!doctype html>
<title>Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
    <input type="text" name="query" placeholder="Enter name, category or keyword" />
    <input type="submit" value="Search" />
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
<a href="/">Back to search</a>
"""

@app.route('/')
def search():
    return render_template_string(search_page)

@app.route('/results')
def results():
    query = request.args.get('query', '').lower()
    matched = []
    for product in products:
        if query in product["name"].lower() or query in product["category"].lower() or query in " ".join(product["keywords"]).lower():
            matched.append(product)
    return render_template_string(results_page, results=matched)

if __name__ == '__main__':
    app.run(debug=True)
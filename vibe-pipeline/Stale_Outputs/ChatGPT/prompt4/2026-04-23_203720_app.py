from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product data
products = [
    {"id": 1, "name": "Red T-shirt", "category": "Clothing", "keywords": "red shirt cotton"},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": "blue denim pants"},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": "mug coffee cup ceramic"},
    {"id": 4, "name": "Laptop Stand", "category": "Electronics", "keywords": "laptop stand metal"},
    {"id": 5, "name": "Wireless Mouse", "category": "Electronics", "keywords": "wireless mouse computer"},
]

search_page = """
<!doctype html>
<title>Online Shop - Search</title>
<h1>Search Products</h1>
<form method="GET" action="/results">
  <input name="query" placeholder="Search by name, category, or keyword" required>
  <button type="submit">Search</button>
</form>
"""

results_page = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
<p>Search query: {{ query }}</p>
{% if results %}
<ul>
  {% for product in results %}
    <li><strong>{{ product['name'] }}</strong> - Category: {{ product['category'] }}</li>
  {% endfor %}
</ul>
{% else %}
<p>No products found matching your search.</p>
{% endif %}
<a href="/">Back to search</a>
"""

@app.route('/')
def search():
    return render_template_string(search_page)

@app.route('/results')
def results():
    query = request.args.get('query', '').lower()
    matching_products = []
    if query:
        for product in products:
            if (query in product['name'].lower() or
                query in product['category'].lower() or
                query in product['keywords'].lower()):
                matching_products.append(product)
    return render_template_string(results_page, query=query, results=matching_products)

if __name__ == '__main__':
    app.run(debug=True)
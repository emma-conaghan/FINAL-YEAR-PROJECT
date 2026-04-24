from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": "red shirt cotton"},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": "blue jeans denim"},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": "coffee mug ceramic"},
    {"id": 4, "name": "Desk Lamp", "category": "Furniture", "keywords": "desk lamp light"},
    {"id": 5, "name": "Running Shoes", "category": "Footwear", "keywords": "running shoes sport"},
]

search_page = """
<!doctype html>
<title>Shop Search</title>
<h1>Search for Products</h1>
<form action="/results" method="get">
    <input type="text" name="query" placeholder="Enter name, category or keyword" required>
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
    results = []
    if query:
        for product in products:
            if (query in product['name'].lower() or
                query in product['category'].lower() or
                query in product['keywords'].lower()):
                results.append(product)
    return render_template_string(results_page, results=results)

if __name__ == '__main__':
    app.run(debug=True)
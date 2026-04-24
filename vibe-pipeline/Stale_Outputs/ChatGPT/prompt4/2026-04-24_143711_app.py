from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"name": "Red T-shirt", "category": "Clothing", "keywords": ["red", "shirt", "casual"]},
    {"name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"name": "Coffee Mug", "category": "Kitchen", "keywords": ["mug", "coffee", "ceramic"]},
    {"name": "Wireless Mouse", "category": "Electronics", "keywords": ["wireless", "mouse", "computer"]},
    {"name": "Running Shoes", "category": "Footwear", "keywords": ["running", "shoes", "sports"]},
    {"name": "Cookbook", "category": "Books", "keywords": ["cookbook", "recipes", "cooking"]},
]

search_page = """
<!doctype html>
<title>Online Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
    <input type="text" name="query" placeholder="Enter name, category or keyword" required>
    <input type="submit" value="Search">
</form>
"""

results_page = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results for '{{ query }}'</h1>
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
    filtered = []
    for product in products:
        if query in product["name"].lower() or query in product["category"].lower() or any(query in kw for kw in product["keywords"]):
            filtered.append(product)
    return render_template_string(results_page, results=filtered, query=query)

if __name__ == '__main__':
    app.run(debug=True)
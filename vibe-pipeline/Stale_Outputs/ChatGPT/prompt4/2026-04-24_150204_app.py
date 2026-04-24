from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product data
PRODUCTS = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": ["red", "shirt", "apparel"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["coffee", "mug", "cup"]},
    {"id": 4, "name": "Desk Lamp", "category": "Electronics", "keywords": ["lamp", "desk", "light"]},
    {"id": 5, "name": "Running Shoes", "category": "Footwear", "keywords": ["running", "shoes", "sport"]},
]

INDEX_HTML = """
<!doctype html>
<title>Shop Search</title>
<h1>Search products</h1>
<form action="/results" method="get">
  <input type="text" name="query" placeholder="Enter name, category or keyword" required>
  <button type="submit">Search</button>
</form>
"""

RESULTS_HTML = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
{% if products %}
<ul>
  {% for product in products %}
  <li>{{ product.name }} (Category: {{ product.category }})</li>
  {% endfor %}
</ul>
{% else %}
<p>No products found.</p>
{% endif %}
<a href="/">New Search</a>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/results")
def results():
    query = request.args.get("query", "").strip().lower()
    matched = []
    if query:
        for product in PRODUCTS:
            name_match = query in product["name"].lower()
            category_match = query in product["category"].lower()
            keyword_match = any(query in keyword for keyword in product["keywords"])
            if name_match or category_match or keyword_match:
                matched.append(product)
    return render_template_string(RESULTS_HTML, products=matched)

if __name__ == "__main__":
    app.run(debug=True)
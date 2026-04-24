from flask import Flask, request, render_template_string

app = Flask(__name__)

# In-memory example product database
products = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": ["red", "shirt", "apparel"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["coffee", "mug", "cup"]},
    {"id": 4, "name": "Desk Lamp", "category": "Electronics", "keywords": ["lamp", "desk", "light"]},
    {"id": 5, "name": "Sneakers", "category": "Footwear", "keywords": ["shoes", "sneakers", "footwear"]},
]

SEARCH_FORM = """
<!doctype html>
<title>Simple Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
  <label for="query">Enter name, category or keyword:</label>
  <input type="text" id="query" name="query" placeholder="Search products...">
  <input type="submit" value="Search">
</form>
"""

RESULTS_PAGE = """
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
    <p>No products found matching your query.</p>
{% endif %}
<a href="/">Back to Search</a>
"""

@app.route("/")
def home():
    return render_template_string(SEARCH_FORM)

@app.route("/results")
def results():
    query = request.args.get("query", "").strip().lower()
    if not query:
        return render_template_string(RESULTS_PAGE, results=[])
    matched = []
    for product in products:
        name = product["name"].lower()
        category = product["category"].lower()
        keywords = [k.lower() for k in product["keywords"]]
        if query in name or query in category or query in keywords:
            matched.append(product)
    return render_template_string(RESULTS_PAGE, results=matched)

if __name__ == "__main__":
    app.run(debug=True)
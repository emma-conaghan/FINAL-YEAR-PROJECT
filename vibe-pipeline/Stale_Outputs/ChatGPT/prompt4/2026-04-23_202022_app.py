from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample in-memory database of products
products = [
    {"name": "Red Shirt", "category": "Clothing", "keywords": ["red", "shirt", "cotton"]},
    {"name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"name": "Coffee Mug", "category": "Kitchen", "keywords": ["coffee", "mug", "ceramic"]},
    {"name": "Running Shoes", "category": "Footwear", "keywords": ["running", "shoes", "sport"]},
    {"name": "Desk Lamp", "category": "Electronics", "keywords": ["lamp", "desk", "light"]},
]

search_page = """
<!doctype html>
<title>Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
  <input type="text" name="query" placeholder="Enter name, category or keyword" required>
  <button type="submit">Search</button>
</form>
"""

results_page = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
<p><a href="/">Back to search</a></p>
{% if results %}
<ul>
  {% for p in results %}
  <li><strong>{{ p.name }}</strong> - Category: {{ p.category }}</li>
  {% endfor %}
</ul>
{% else %}
<p>No products found.</p>
{% endif %}
"""

@app.route("/")
def search():
    return render_template_string(search_page)

@app.route("/results")
def results():
    query = request.args.get("query", "").lower()
    results = []
    if query:
        for product in products:
            if (query in product["name"].lower() or
                query in product["category"].lower() or
                any(query in k for k in product["keywords"])):
                results.append(product)
    return render_template_string(results_page, results=results)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product data
products = [
    {"id": 1, "name": "Red T-Shirt", "category": "Clothing", "keywords": ["red", "shirt", "cotton"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Accessories", "keywords": ["coffee", "mug", "ceramic"]},
    {"id": 4, "name": "Running Shoes", "category": "Footwear", "keywords": ["running", "shoes", "sport"]},
    {"id": 5, "name": "Smartphone", "category": "Electronics", "keywords": ["phone", "smartphone", "mobile"]},
]

SEARCH_PAGE = """
<!doctype html>
<title>Online Shop - Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
  <input type="text" name="query" placeholder="Search by name, category or keyword" size="40">
  <input type="submit" value="Search">
</form>
"""

RESULTS_PAGE = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
{% if results %}
<ul>
  {% for p in results %}
    <li><strong>{{ p.name }}</strong> (Category: {{ p.category }})</li>
  {% endfor %}
</ul>
{% else %}
<p>No products found matching your search.</p>
{% endif %}
<a href="/">Back to search</a>
"""

@app.route("/")
def search():
    return render_template_string(SEARCH_PAGE)

@app.route("/results")
def results():
    query = request.args.get("query", "").strip().lower()
    if not query:
        matched = []
    else:
        matched = []
        for product in products:
            if (query in product["name"].lower() or
                query in product["category"].lower() or
                any(query in kw for kw in product["keywords"])):
                matched.append(product)
    return render_template_string(RESULTS_PAGE, results=matched)

if __name__ == "__main__":
    app.run(debug=True)
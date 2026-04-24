from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample in-memory product database
products = [
    {"id": 1, "name": "Blue T-shirt", "category": "Clothing", "keywords": ["blue", "shirt", "cotton"]},
    {"id": 2, "name": "Red Sneakers", "category": "Footwear", "keywords": ["red", "shoes", "running"]},
    {"id": 3, "name": "Coffee Mug", "category": "Home", "keywords": ["coffee", "mug", "ceramic"]},
    {"id": 4, "name": "Wireless Mouse", "category": "Electronics", "keywords": ["wireless", "mouse", "computer"]},
    {"id": 5, "name": "Green Hoodie", "category": "Clothing", "keywords": ["green", "hoodie", "cotton"]},
]

search_page = """
<!doctype html>
<title>Search Products</title>
<h1>Search for Products</h1>
<form action="/results" method="get">
  <label for="q">Search by name:</label><br>
  <input type="text" id="q" name="q"><br><br>
  <label for="category">Category:</label><br>
  <input type="text" id="category" name="category"><br><br>
  <label for="keyword">Keyword:</label><br>
  <input type="text" id="keyword" name="keyword"><br><br>
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
    <li>{{ product.name }} - Category: {{ product.category }}</li>
  {% endfor %}
  </ul>
{% else %}
  <p>No products found matching your search.</p>
{% endif %}
<a href="/">Back to search</a>
"""

@app.route("/")
def search():
    return render_template_string(search_page)

@app.route("/results")
def results():
    q = request.args.get("q", "").lower()
    category = request.args.get("category", "").lower()
    keyword = request.args.get("keyword", "").lower()

    filtered = []
    for product in products:
        name_match = q in product["name"].lower() if q else True
        category_match = category in product["category"].lower() if category else True
        keyword_match = any(keyword in k for k in product["keywords"]) if keyword else True

        if name_match and category_match and keyword_match:
            filtered.append(product)

    return render_template_string(results_page, results=filtered)

if __name__ == "__main__":
    app.run(debug=True)
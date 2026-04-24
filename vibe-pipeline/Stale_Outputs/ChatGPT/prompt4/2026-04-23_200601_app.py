from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": ["red", "shirt", "cotton"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["coffee", "mug", "cup"]},
    {"id": 4, "name": "Headphones", "category": "Electronics", "keywords": ["headphones", "audio", "music"]},
    {"id": 5, "name": "Running Shoes", "category": "Footwear", "keywords": ["shoes", "running", "sport"]},
]

search_form = """
<!doctype html>
<title>Simple Shop Search</title>
<h1>Search Products</h1>
<form action="/search" method="get">
    <input type="text" name="query" placeholder="Enter name, category or keyword" required>
    <button type="submit">Search</button>
</form>
"""

results_page = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results for "{{ query }}"</h1>
{% if results %}
<ul>
  {% for product in results %}
    <li>{{ product.name }} - Category: {{ product.category }}</li>
  {% endfor %}
</ul>
{% else %}
<p>No products found.</p>
{% endif %}
<p><a href="/">Back to search</a></p>
"""

@app.route("/")
def index():
    return search_form

@app.route("/search")
def search():
    query = request.args.get("query", "").strip().lower()
    results = []
    if query:
        for product in products:
            if (query in product["name"].lower() or
                query in product["category"].lower() or
                query in " ".join(product["keywords"]).lower()):
                results.append(product)
    return render_template_string(results_page, query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
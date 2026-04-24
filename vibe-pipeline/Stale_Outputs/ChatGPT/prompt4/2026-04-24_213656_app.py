from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"id": 1, "name": "Red T-Shirt", "category": "Clothing", "keywords": ["red", "shirt", "cotton"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["coffee", "mug", "ceramic"]},
    {"id": 4, "name": "Laptop Stand", "category": "Electronics", "keywords": ["laptop", "stand", "office"]},
    {"id": 5, "name": "Running Shoes", "category": "Footwear", "keywords": ["running", "shoes", "sport"]},
]

search_page = """
<!doctype html>
<html>
<head><title>Online Shop Search</title></head>
<body>
<h1>Search Products</h1>
<form method="get" action="/results">
    <input type="text" name="query" placeholder="Search by name, category, or keyword" size="40" required>
    <input type="submit" value="Search">
</form>
</body>
</html>
"""

results_page = """
<!doctype html>
<html>
<head><title>Search Results</title></head>
<body>
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
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(search_page)

@app.route("/results")
def results():
    query = request.args.get("query", "").strip().lower()
    if not query:
        return render_template_string(results_page, results=[])
    matched = []
    for product in products:
        if (query in product["name"].lower()
                or query in product["category"].lower()
                or any(query in kw.lower() for kw in product["keywords"])):
            matched.append(product)
    return render_template_string(results_page, results=matched)

if __name__ == "__main__":
    app.run(debug=True)
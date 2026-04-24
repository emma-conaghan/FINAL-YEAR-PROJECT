from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product data
products = [
    {"id": 1, "name": "Red T-shirt", "category": "Clothing", "keywords": ["red", "shirt", "tshirt", "cotton"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Running Shoes", "category": "Footwear", "keywords": ["shoes", "running", "sport"]},
    {"id": 4, "name": "Coffee Mug", "category": "Kitchenware", "keywords": ["coffee", "mug", "cup"]},
    {"id": 5, "name": "Desk Lamp", "category": "Electronics", "keywords": ["lamp", "desk", "light"]},
]

# Simple homepage with search form
home_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Online Shop</title>
</head>
<body>
    <h1>Online Shop</h1>
    <form action="/search" method="get">
        <input type="text" name="query" placeholder="Search by name, category, or keyword" size="40">
        <input type="submit" value="Search">
    </form>
</body>
</html>
"""

# Results page template
results_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Search Results</title>
</head>
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
    <p><a href="/">Back to search</a></p>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(home_html)

@app.route("/search")
def search():
    query = request.args.get("query", "").strip().lower()
    matching_products = []
    if query:
        for product in products:
            # Search name, category, and keywords for the query string
            if query in product["name"].lower() or query in product["category"].lower() or any(query in k for k in product["keywords"]):
                matching_products.append(product)
    return render_template_string(results_html, results=matching_products)

if __name__ == "__main__":
    app.run(debug=True)
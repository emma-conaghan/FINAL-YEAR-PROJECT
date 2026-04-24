from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": ["red", "shirt", "cotton"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["coffee", "mug", "ceramic"]},
    {"id": 4, "name": "Wireless Mouse", "category": "Electronics", "keywords": ["wireless", "mouse", "computer"]},
    {"id": 5, "name": "Cooking Book", "category": "Books", "keywords": ["cooking", "book", "recipes"]},
]

search_page = """
<!doctype html>
<html>
    <head>
        <title>Simple Shop - Search</title>
    </head>
    <body>
        <h1>Search Products</h1>
        <form action="/results" method="get">
            <input type="text" name="q" placeholder="Search by name, category, or keyword" size="40" required>
            <button type="submit">Search</button>
        </form>
    </body>
</html>
"""

results_page = """
<!doctype html>
<html>
    <head>
        <title>Simple Shop - Results</title>
    </head>
    <body>
        <h1>Search Results</h1>
        {% if results %}
            <ul>
            {% for p in results %}
                <li><strong>{{ p.name }}</strong> - Category: {{ p.category }}</li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No products found.</p>
        {% endif %}
        <a href="/">Back to search</a>
    </body>
</html>
"""

@app.route("/")
def search():
    return render_template_string(search_page)

@app.route("/results")
def results():
    query = request.args.get("q", "").strip().lower()
    matched = []
    if query:
        for product in products:
            if (query in product["name"].lower()
                or query in product["category"].lower()
                or any(query in keyword for keyword in product["keywords"])):
                matched.append(product)
    return render_template_string(results_page, results=matched)

if __name__ == "__main__":
    app.run(debug=True)
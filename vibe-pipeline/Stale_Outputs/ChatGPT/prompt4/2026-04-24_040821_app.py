from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"id": 1, "name": "Red T-Shirt", "category": "Clothing", "keywords": "red shirt casual"},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": "blue jeans denim"},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": "coffee mug cup"},
    {"id": 4, "name": "Gaming Mouse", "category": "Electronics", "keywords": "gaming mouse pc"},
    {"id": 5, "name": "Sneakers", "category": "Footwear", "keywords": "shoes sneakers sport"},
]

search_form_html = """
<!doctype html>
<title>Simple Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
    <input type="text" name="query" placeholder="Search by name, category or keyword">
    <input type="submit" value="Search">
</form>
"""

results_page_html = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
<form action="/results" method="get">
    <input type="text" name="query" placeholder="Search by name, category or keyword" value="{{query}}">
    <input type="submit" value="Search">
</form>
{% if results %}
<ul>
    {% for product in results %}
    <li><strong>{{product.name}}</strong> - Category: {{product.category}}</li>
    {% endfor %}
</ul>
{% else %}
<p>No products found.</p>
{% endif %}
<a href="/">Back to Search</a>
"""

@app.route("/")
def search():
    return render_template_string(search_form_html)

@app.route("/results")
def results():
    query = request.args.get("query", "").strip().lower()
    filtered = []
    if query:
        for product in products:
            name = product["name"].lower()
            category = product["category"].lower()
            keywords = product["keywords"].lower()
            if query in name or query in category or query in keywords:
                filtered.append(product)
    return render_template_string(results_page_html, results=filtered, query=query)

if __name__ == "__main__":
    app.run(debug=True)
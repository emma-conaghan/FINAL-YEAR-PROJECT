from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product data
products = [
    {"id": 1, "name": "Red T-Shirt", "category": "Clothing", "keywords": ["red", "shirt", "summer"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "denim", "pants"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["mug", "coffee", "cup"]},
    {"id": 4, "name": "Sneakers", "category": "Footwear", "keywords": ["shoes", "sports", "running"]},
    {"id": 5, "name": "Laptop", "category": "Electronics", "keywords": ["computer", "portable", "work"]},
]

search_form_html = """
<!doctype html>
<title>Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
  <input type="text" name="query" placeholder="Enter name, category, or keyword" required>
  <button type="submit">Search</button>
</form>
"""

results_html = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
<p>Search query: {{ query }}</p>
{% if results %}
  <ul>
  {% for p in results %}
    <li><strong>{{ p['name'] }}</strong> - Category: {{ p['category'] }}</li>
  {% endfor %}
  </ul>
{% else %}
  <p>No products found.</p>
{% endif %}
<br>
<a href="/">Back to search</a>
"""

@app.route("/")
def search():
    return search_form_html

@app.route("/results")
def results():
    query = request.args.get("query", "").strip().lower()
    results = []
    if query:
        for p in products:
            if (query in p["name"].lower()
                or query in p["category"].lower()
                or any(query in kw for kw in p["keywords"])):
                results.append(p)
    return render_template_string(results_html, query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample data to simulate a product database
products = [
    {"id": 1, "name": "Red T-Shirt", "category": "Clothing", "keywords": ["red", "shirt", "apparel"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["coffee", "mug", "cup"]},
    {"id": 4, "name": "Book: Python Basics", "category": "Books", "keywords": ["python", "programming", "book"]},
    {"id": 5, "name": "Wireless Mouse", "category": "Electronics", "keywords": ["wireless", "mouse", "computer"]},
]

search_form_html = '''
<!doctype html>
<title>Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
  <label for="query">Search by name, category, or keyword:</label><br>
  <input type="text" id="query" name="query" placeholder="Enter search term">
  <button type="submit">Search</button>
</form>
'''

results_html = '''
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
  <p>No products found matching your search.</p>
{% endif %}
<a href="/">Back to search</a>
'''

@app.route("/")
def home():
    return render_template_string(search_form_html)

@app.route("/results")
def results():
    query = request.args.get("query", "").strip().lower()
    if not query:
        filtered = []
    else:
        filtered = []
        for p in products:
            if (query in p["name"].lower() or
                query in p["category"].lower() or
                any(query in kw for kw in p["keywords"])):
                filtered.append(p)

    return render_template_string(results_html, results=filtered)

if __name__ == "__main__":
    app.run(debug=True)
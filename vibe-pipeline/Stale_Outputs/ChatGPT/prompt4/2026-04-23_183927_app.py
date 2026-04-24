from flask import Flask, request, render_template_string

app = Flask(__name__)

# Example product database
products = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": ["red", "shirt", "cotton"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["mug", "coffee", "ceramic"]},
    {"id": 4, "name": "Gaming Mouse", "category": "Electronics", "keywords": ["mouse", "gaming", "usb"]},
    {"id": 5, "name": "Notebook", "category": "Stationery", "keywords": ["notebook", "paper", "writing"]},
]

SEARCH_HTML = """
<!doctype html>
<title>Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
  <input type="text" name="search" placeholder="Enter name, category or keyword">
  <input type="submit" value="Search">
</form>
"""

RESULTS_HTML = """
<!doctype html>
<title>Search Results</title>
<h1>Search Results</h1>
<p><a href="/">Back to Search</a></p>
{% if results %}
  <ul>
  {% for p in results %}
    <li><strong>{{ p['name'] }}</strong> (Category: {{ p['category'] }})</li>
  {% endfor %}
  </ul>
{% else %}
  <p>No products found.</p>
{% endif %}
"""

@app.route("/")
def search():
    return render_template_string(SEARCH_HTML)

@app.route("/results")
def results():
    query = request.args.get("search", "").strip().lower()
    if query == "":
        filtered = []
    else:
        filtered = []
        for p in products:
            if query in p["name"].lower():
                filtered.append(p)
                continue
            if query in p["category"].lower():
                filtered.append(p)
                continue
            for kw in p["keywords"]:
                if query == kw.lower():
                    filtered.append(p)
                    break
    return render_template_string(RESULTS_HTML, results=filtered)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product data
products = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": "red,shirt,clothing,fashion"},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": "blue,jeans,clothing,fashion"},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": "coffee,mug,kitchen,drink"},
    {"id": 4, "name": "Bluetooth Speaker", "category": "Electronics", "keywords": "bluetooth,speaker,electronics,music"},
    {"id": 5, "name": "Notebook", "category": "Stationery", "keywords": "notebook,stationery,writing,paper"},
]

search_page = """
<!doctype html>
<title>Online Shop Search</title>
<h1>Search for Products</h1>
<form action="/results" method="get">
    <label for="query">Search by name, category, or keyword:</label><br>
    <input type="text" id="query" name="query" size="40">
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
        <li><b>{{ product['name'] }}</b> - Category: {{ product['category'] }}</li>
    {% endfor %}
    </ul>
{% else %}
    <p>No products found matching your query.</p>
{% endif %}
<p><a href="/">Back to search</a></p>
"""

@app.route('/')
def search():
    return render_template_string(search_page)

@app.route('/results')
def results():
    query = request.args.get("query", "").lower().strip()
    if not query:
        return render_template_string(results_page, results=[])
    matched = []
    for product in products:
        if query in product["name"].lower() or query in product["category"].lower() or query in product["keywords"].lower():
            matched.append(product)
    return render_template_string(results_page, results=matched)

if __name__ == '__main__':
    app.run(debug=True)
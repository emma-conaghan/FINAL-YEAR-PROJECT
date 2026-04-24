from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"id": 1, "name": "Red Shirt", "category": "Clothing", "keywords": ["red", "shirt", "apparel"]},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": ["blue", "jeans", "denim"]},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchen", "keywords": ["mug", "coffee", "cup"]},
    {"id": 4, "name": "Notebook", "category": "Stationery", "keywords": ["notebook", "paper", "writing"]},
    {"id": 5, "name": "Headphones", "category": "Electronics", "keywords": ["headphones", "audio", "music"]}
]

search_page = '''
<!doctype html>
<title>Shop Search</title>
<h1>Search Products</h1>
<form action="/results" method="get">
  <input type="text" name="query" placeholder="Search by name, category, or keyword" size="40" required>
  <input type="submit" value="Search">
</form>
'''

results_page = '''
<!doctype html>
<title>Search Results</title>
<h1>Search Results for "{{ query }}"</h1>
{% if results %}
  <ul>
  {% for product in results %}
    <li>{{ product["name"] }} - Category: {{ product["category"] }}</li>
  {% endfor %}
  </ul>
{% else %}
  <p>No products found.</p>
{% endif %}
<p><a href="/">Back to Search</a></p>
'''

@app.route('/')
def search():
    return render_template_string(search_page)

@app.route('/results')
def results():
    query = request.args.get('query', '').strip().lower()
    matched = []
    if query:
        for product in products:
            if query in product["name"].lower() or query in product["category"].lower() or any(query in kw for kw in product["keywords"]):
                matched.append(product)
    return render_template_string(results_page, query=query, results=matched)

if __name__ == '__main__':
    app.run(debug=True)
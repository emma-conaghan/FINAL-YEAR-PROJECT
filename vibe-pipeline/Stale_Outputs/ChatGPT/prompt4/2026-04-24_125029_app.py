from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample product database
products = [
    {"id": 1, "name": "Red T-Shirt", "category": "Clothing", "keywords": "red tshirt cotton"},
    {"id": 2, "name": "Blue Jeans", "category": "Clothing", "keywords": "blue jeans denim"},
    {"id": 3, "name": "Coffee Mug", "category": "Kitchenware", "keywords": "mug coffee cup"},
    {"id": 4, "name": "Laptop", "category": "Electronics", "keywords": "computer laptop electronics"},
    {"id": 5, "name": "Headphones", "category": "Electronics", "keywords": "sound music headphones"}
]

search_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Simple Shop Search</title>
</head>
<body>
    <h1>Search Products</h1>
    <form action="/results" method="get">
        <input type="text" name="query" placeholder="Search by name, category, or keyword" required>
        <button type="submit">Search</button>
    </form>
</body>
</html>
'''

results_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Search Results</title>
</head>
<body>
    <h1>Search Results for "{{ query }}"</h1>
    {% if results %}
        <ul>
        {% for product in results %}
            <li><strong>{{ product.name }}</strong> - Category: {{ product.category }}</li>
        {% endfor %}
        </ul>
    {% else %}
        <p>No products found.</p>
    {% endif %}
    <a href="/">Back to Search</a>
</body>
</html>
'''

@app.route('/')
def search():
    return render_template_string(search_page)

@app.route('/results')
def results():
    query = request.args.get('query', '').strip().lower()
    matches = []
    if query:
        for p in products:
            if query in p['name'].lower() or query in p['category'].lower() or query in p['keywords'].lower():
                matches.append(p)
    return render_template_string(results_page, query=query, results=matches)

if __name__ == '__main__':
    app.run(debug=True)
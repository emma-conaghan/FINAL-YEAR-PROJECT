from flask import Flask, request, render_template_string

app = Flask(__name__)

# Sample data representing a database
PRODUCTS = [
    {"id": 1, "name": "Modern Coffee Mug", "category": "Kitchen", "keywords": "ceramic, drinkware, cup"},
    {"id": 2, "name": "Wireless Gaming Mouse", "category": "Electronics", "keywords": "computer, gaming, peripheral"},
    {"id": 3, "name": "Organic Cotton T-Shirt", "category": "Apparel", "keywords": "clothing, shirt, eco-friendly"},
    {"id": 4, "name": "LED Desk Lamp", "category": "Home Office", "keywords": "lighting, study, furniture"},
    {"id": 5, "name": "Bluetooth Headphones", "category": "Electronics", "keywords": "audio, music, wireless"},
    {"id": 6, "name": "Stainless Steel Water Bottle", "category": "Kitchen", "keywords": "gym, hydration, flask"}
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Simple Online Shop</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px; background-color: #f4f4f4; }
        .container { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .search-area { background: #eef; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        input[type="text"] { padding: 10px; width: 70%; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #218838; }
        .product-card { border-bottom: 1px solid #eee; padding: 15px 0; }
        .product-card:last-child { border-bottom: none; }
        .category { color: #666; font-size: 0.9em; font-style: italic; }
        .tags { color: #888; font-size: 0.8em; }
        .no-results { color: #d9534f; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Simple Online Shop</h1>
        
        <div class="search-area">
            <form action="/search" method="get">
                <input type="text" name="query" placeholder="Search by name, category, or tag..." value="{{ search_term }}">
                <button type="submit">Search</button>
            </form>
        </div>

        {% if search_term %}
            <h3>Results for: "{{ search_term }}"</h3>
            {% if products %}
                {% for item in products %}
                <div class="product-card">
                    <strong>{{ item.name }}</strong><br>
                    <span class="category">Category: {{ item.category }}</span><br>
                    <span class="tags">Keywords: {{ item.keywords }}</span>
                </div>
                {% endfor %}
            {% else %}
                <p class="no-results">No products found matching your search.</p>
            {% endif %}
            <p><a href="/">View All Products</a></p>
        {% else %}
            <h3>Featured Products</h3>
            {% for item in products %}
            <div class="product-card">
                <strong>{{ item.name }}</strong><br>
                <span class="category">Category: {{ item.category }}</span>
            </div>
            {% endfor %}
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    # Show all products on the home page
    return render_template_string(HTML_TEMPLATE, products=PRODUCTS, search_term="")

@app.route('/search')
def search():
    query = request.args.get('query', '').strip().lower()
    
    if not query:
        return render_template_string(HTML_TEMPLATE, products=PRODUCTS, search_term="")

    # Filter products based on name, category, or keywords
    filtered_results = []
    for product in PRODUCTS:
        if (query in product['name'].lower() or 
            query in product['category'].lower() or 
            query in product['keywords'].lower()):
            filtered_results.append(product)

    return render_template_string(HTML_TEMPLATE, products=filtered_results, search_term=query)

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
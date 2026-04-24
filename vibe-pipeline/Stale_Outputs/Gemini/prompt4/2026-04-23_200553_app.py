from flask import Flask, request, render_template_string

app = Flask(__name__)

# Mock database of products
PRODUCTS = [
    {"name": "Smartphone X1", "category": "Electronics", "keywords": "mobile, phone, technology"},
    {"name": "Ceramic Coffee Mug", "category": "Kitchen", "keywords": "drink, cup, home"},
    {"name": "Ultra-light Running Shoes", "category": "Footwear", "keywords": "sports, fitness, exercise"},
    {"name": "Mechanical Keyboard", "category": "Electronics", "keywords": "computer, typing, peripheral"},
    {"name": "Stainless Steel Water Bottle", "category": "Accessories", "keywords": "gym, travel, hydration"},
    {"name": "History of Art Book", "category": "Books", "keywords": "education, reading, culture"},
    {"name": "Cotton T-Shirt", "category": "Apparel", "keywords": "clothing, fashion, basics"}
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Shop</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6; color: #333; }
        .search-container { background: #f0f0f0; padding: 25px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
        .product-list { display: grid; gap: 15px; }
        .product-card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; transition: transform 0.2s; }
        .product-card:hover { border-color: #aaa; background-color: #fafafa; }
        .category-tag { background: #e2e2e2; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; color: #555; }
        input[type="text"] { padding: 10px; width: 300px; border: 1px solid #ccc; border-radius: 4px; }
        button { padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #218838; }
        .back-link { display: inline-block; margin-top: 20px; text-decoration: none; color: #007bff; }
    </style>
</head>
<body>
    <h1>Local Online Shop</h1>
    
    <div class="search-container">
        <form action="/search" method="get">
            <input type="text" name="q" placeholder="Search name, category, or keyword..." value="{{ query or '' }}">
            <button type="submit">Search</button>
        </form>
    </div>

    {% if query %}
        <h2>Results for "{{ query }}"</h2>
        <div class="product-list">
            {% if results %}
                {% for item in results %}
                <div class="product-card">
                    <h3>{{ item.name }}</h3>
                    <p><span class="category-tag">{{ item.category }}</span></p>
                    <small>Tags: {{ item.keywords }}</small>
                </div>
                {% endfor %}
            {% else %}
                <p>Sorry, no items matched your search criteria.</p>
            {% endif %}
        </div>
        <a href="/" class="back-link">← Show all products</a>
    {% else %}
        <h2>Featured Products</h2>
        <div class="product-list">
            {% for item in all_products %}
            <div class="product-card">
                <h3>{{ item.name }}</h3>
                <p><span class="category-tag">{{ item.category }}</span></p>
            </div>
            {% endfor %}
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, all_products=PRODUCTS, query=None)

@app.route('/search')
def search_products():
    search_query = request.args.get('q', '').lower()
    matches = []
    
    if search_query:
        for product in PRODUCTS:
            name_match = search_query in product['name'].lower()
            category_match = search_query in product['category'].lower()
            keyword_match = search_query in product['keywords'].lower()
            
            if name_match or category_match or keyword_match:
                matches.append(product)
                
    return render_template_string(
        HTML_TEMPLATE, 
        results=matches, 
        query=search_query, 
        all_products=PRODUCTS
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
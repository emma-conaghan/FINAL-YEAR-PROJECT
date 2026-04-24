import http.client
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote


API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")


def get_weather(city):
    if not API_KEY:
        return {"error": "No API key set. Please set the OPENWEATHERMAP_API_KEY environment variable."}
    
    try:
        conn = http.client.HTTPSConnection("api.openweathermap.org", timeout=10)
        encoded_city = quote(city)
        url = f"/data/2.5/weather?q={encoded_city}&appid={API_KEY}&units=metric"
        conn.request("GET", url)
        response = conn.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        conn.close()
        
        if response.status != 200:
            return {"error": data.get("message", "Unknown error from weather API")}
        
        result = {
            "city": data.get("name", city),
            "country": data.get("sys", {}).get("country", "N/A"),
            "temperature": data.get("main", {}).get("temp", "N/A"),
            "feels_like": data.get("main", {}).get("feels_like", "N/A"),
            "humidity": data.get("main", {}).get("humidity", "N/A"),
            "pressure": data.get("main", {}).get("pressure", "N/A"),
            "wind_speed": data.get("wind", {}).get("speed", "N/A"),
            "description": data.get("weather", [{}])[0].get("description", "N/A"),
            "icon": data.get("weather", [{}])[0].get("icon", "01d"),
        }
        return result
    except Exception as e:
        return {"error": str(e)}


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weather App</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 480px;
            width: 90%;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            border-color: #667eea;
        }
        button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: opacity 0.3s;
        }
        button:hover {
            opacity: 0.9;
        }
        .weather-result {
            display: none;
        }
        .weather-result.active {
            display: block;
        }
        .weather-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .city-name {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .weather-icon {
            width: 100px;
            height: 100px;
        }
        .temperature {
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
        }
        .description {
            font-size: 18px;
            color: #666;
            text-transform: capitalize;
            margin-top: 5px;
        }
        .weather-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 25px;
        }
        .detail-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .detail-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .detail-value {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        .error-message {
            background: #ffe0e0;
            color: #d32f2f;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            display: none;
        }
        .error-message.active {
            display: block;
        }
        .loading {
            text-align: center;
            color: #666;
            display: none;
        }
        .loading.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤 Weather App</h1>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." onkeypress="if(event.key==='Enter')searchWeather()">
            <button onclick="searchWeather()">Search</button>
        </div>
        <div class="loading" id="loading">Loading...</div>
        <div class="error-message" id="error"></div>
        <div class="weather-result" id="weatherResult">
            <div class="weather-header">
                <div class="city-name" id="cityName"></div>
                <img class="weather-icon" id="weatherIcon" src="" alt="weather icon">
                <div class="temperature" id="temperature"></div>
                <div class="description" id="description"></div>
            </div>
            <div class="weather-details">
                <div class="detail-card">
                    <div class="detail-label">Feels Like</div>
                    <div class="detail-value" id="feelsLike"></div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Humidity</div>
                    <div class="detail-value" id="humidity"></div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Wind Speed</div>
                    <div class="detail-value" id="windSpeed"></div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Pressure</div>
                    <div class="detail-value" id="pressure"></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        async function searchWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) return;

            document.getElementById('loading').classList.add('active');
            document.getElementById('weatherResult').classList.remove('active');
            document.getElementById('error').classList.remove('active');

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                document.getElementById('loading').classList.remove('active');

                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').classList.add('active');
                    return;
                }

                document.getElementById('cityName').textContent = data.city + ', ' + data.country;
                document.getElementById('weatherIcon').src = 'https://openweathermap.org/img/wn/' + data.icon + '@2x.png';
                document.getElementById('temperature').textContent = Math.round(data.temperature) + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = Math.round(data.feels_like) + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('windSpeed').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';
                document.getElementById('weatherResult').classList.add('active');
            } catch (err) {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('error').textContent = 'Failed to fetch weather data. Please try again.';
                document.getElementById('error').classList.add('active');
            }
        }
    </script>
</body>
</html>"""


class WeatherHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        
        elif parsed.path == "/weather":
            params = parse_qs(parsed.query)
            city = params.get("city", [""])[0]
            
            if not city:
                result = {"error": "Please enter a city name."}
            else:
                result = get_weather(city)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0] if args else ''}")


def main():
    port = int(os.environ.get("PORT", 8000))
    
    if not API_KEY:
        print("=" * 60)
        print("WARNING: No OpenWeatherMap API key found!")
        print("Set it with: export OPENWEATHERMAP_API_KEY=your_key_here")
        print("Get a free key at: https://openweathermap.org/api")
        print("=" * 60)
    else:
        print(f"API key loaded (ends with ...{API_KEY[-4:]})")
    
    server = HTTPServer(("0.0.0.0", port), WeatherHandler)
    print(f"Weather App running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
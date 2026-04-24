import http.client
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

API_KEY = "demo"
BASE_HOST = "api.openweathermap.org"


def get_weather(city):
    """Fetch weather data from OpenWeatherMap API."""
    params = urllib.parse.urlencode({
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    })
    url = f"/data/2.5/weather?{params}"

    try:
        conn = http.client.HTTPSConnection(BASE_HOST, timeout=10)
        conn.request("GET", url)
        response = conn.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        conn.close()

        if response.status == 200:
            return {
                "success": True,
                "city": data.get("name", city),
                "country": data.get("sys", {}).get("country", "N/A"),
                "temperature": data.get("main", {}).get("temp", "N/A"),
                "feels_like": data.get("main", {}).get("feels_like", "N/A"),
                "humidity": data.get("main", {}).get("humidity", "N/A"),
                "pressure": data.get("main", {}).get("pressure", "N/A"),
                "description": data.get("weather", [{}])[0].get("description", "N/A"),
                "wind_speed": data.get("wind", {}).get("speed", "N/A"),
                "icon": data.get("weather", [{}])[0].get("icon", "01d"),
            }
        else:
            message = data.get("message", "Unknown error")
            return {"success": False, "error": f"API Error ({response.status}): {message}"}
    except Exception as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}


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
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            max-width: 500px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
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
            padding: 12px 20px;
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
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
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
            color: #333;
            font-weight: bold;
        }
        .country {
            color: #888;
            font-size: 14px;
        }
        .weather-icon {
            width: 100px;
            height: 100px;
        }
        .temperature {
            font-size: 48px;
            font-weight: bold;
            color: #333;
        }
        .description {
            text-transform: capitalize;
            color: #666;
            font-size: 18px;
            margin-top: 5px;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 25px;
            padding-top: 25px;
            border-top: 1px solid #eee;
        }
        .detail-item {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 10px;
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
        .error {
            text-align: center;
            color: #e74c3c;
            padding: 20px;
            background: #ffeaea;
            border-radius: 10px;
            display: none;
        }
        .error.active {
            display: block;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
            display: none;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            display: inline-block;
            width: 30px;
            height: 30px;
            border: 3px solid #667eea;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .api-note {
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: #999;
        }
        .api-note a {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤 Weather App</h1>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." autofocus>
            <button onclick="getWeather()">Search</button>
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px;">Fetching weather data...</p>
        </div>
        <div class="error" id="error"></div>
        <div class="weather-result" id="weatherResult">
            <div class="weather-header">
                <div class="city-name" id="cityName"></div>
                <div class="country" id="country"></div>
                <img class="weather-icon" id="weatherIcon" src="" alt="Weather icon">
                <div class="temperature" id="temperature"></div>
                <div class="description" id="description"></div>
            </div>
            <div class="details">
                <div class="detail-item">
                    <div class="detail-label">Feels Like</div>
                    <div class="detail-value" id="feelsLike"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Humidity</div>
                    <div class="detail-value" id="humidity"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Wind Speed</div>
                    <div class="detail-value" id="windSpeed"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Pressure</div>
                    <div class="detail-value" id="pressure"></div>
                </div>
            </div>
        </div>
        <div class="api-note">
            Powered by <a href="https://openweathermap.org/" target="_blank">OpenWeatherMap</a><br>
            Set your API key in app.py (get one free at openweathermap.org)
        </div>
    </div>
    <script>
        document.getElementById('cityInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') getWeather();
        });

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) return;

            document.getElementById('loading').classList.add('active');
            document.getElementById('weatherResult').classList.remove('active');
            document.getElementById('error').classList.remove('active');

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                document.getElementById('loading').classList.remove('active');

                if (data.success) {
                    document.getElementById('cityName').textContent = data.city;
                    document.getElementById('country').textContent = data.country;
                    document.getElementById('weatherIcon').src =
                        'https://openweathermap.org/img/wn/' + data.icon + '@2x.png';
                    document.getElementById('temperature').textContent =
                        Math.round(data.temperature) + '°C';
                    document.getElementById('description').textContent = data.description;
                    document.getElementById('feelsLike').textContent =
                        Math.round(data.feels_like) + '°C';
                    document.getElementById('humidity').textContent = data.humidity + '%';
                    document.getElementById('windSpeed').textContent = data.wind_speed + ' m/s';
                    document.getElementById('pressure').textContent = data.pressure + ' hPa';
                    document.getElementById('weatherResult').classList.add('active');
                } else {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').classList.add('active');
                }
            } catch (err) {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('error').textContent = 'Failed to connect to server: ' + err.message;
                document.getElementById('error').classList.add('active');
            }
        }
    </script>
</body>
</html>"""


class WeatherHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path.startswith("/weather"):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            city = params.get("city", [""])[0]

            if not city:
                result = {"success": False, "error": "Please enter a city name."}
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
    host = "localhost"
    port = 8000

    if API_KEY == "demo":
        print("=" * 60)
        print("  WARNING: Using demo API key!")
        print("  For full functionality, get a free API key at:")
        print("  https://openweathermap.org/appid")
        print("  Then set API_KEY in app.py")
        print("=" * 60)

    print(f"\n🌤  Weather App is running!")
    print(f"   Open your browser at: http://{host}:{port}\n")

    server = HTTPServer((host, port), WeatherHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
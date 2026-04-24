import http.server
import socketserver
import json
import urllib.request
import urllib.parse
import urllib.error
import os

PORT = 8000

API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")

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
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 500px;
            width: 90%;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2em;
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
        .weather-info {
            display: none;
            text-align: center;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .city-name {
            font-size: 1.8em;
            color: #333;
            margin-bottom: 5px;
        }
        .country {
            font-size: 1em;
            color: #888;
            margin-bottom: 15px;
        }
        .weather-icon {
            width: 100px;
            height: 100px;
        }
        .temperature {
            font-size: 3em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }
        .description {
            font-size: 1.2em;
            color: #666;
            text-transform: capitalize;
            margin-bottom: 20px;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .detail-item {
            text-align: center;
        }
        .detail-label {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 5px;
        }
        .detail-value {
            font-size: 1.1em;
            color: #333;
            font-weight: 600;
        }
        .error {
            color: #e74c3c;
            text-align: center;
            font-size: 1.1em;
            display: none;
            padding: 15px;
            background: #ffeaea;
            border-radius: 10px;
        }
        .loading {
            text-align: center;
            color: #667eea;
            font-size: 1.1em;
            display: none;
        }
        .api-note {
            text-align: center;
            color: #999;
            font-size: 0.8em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>&#9925; Weather App</h1>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." onkeypress="if(event.key==='Enter')getWeather()">
            <button onclick="getWeather()">Search</button>
        </div>
        <div class="loading" id="loading">Loading...</div>
        <div class="error" id="error"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <div class="country" id="country"></div>
            <img class="weather-icon" id="weatherIcon" src="" alt="weather icon">
            <div class="temperature" id="temperature"></div>
            <div class="description" id="description"></div>
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
                    <div class="detail-value" id="wind"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Pressure</div>
                    <div class="detail-value" id="pressure"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Visibility</div>
                    <div class="detail-value" id="visibility"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Clouds</div>
                    <div class="detail-value" id="clouds"></div>
                </div>
            </div>
        </div>
        <div class="api-note">Powered by OpenWeatherMap API</div>
    </div>

    <script>
        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) return;

            const weatherInfo = document.getElementById('weatherInfo');
            const error = document.getElementById('error');
            const loading = document.getElementById('loading');

            weatherInfo.style.display = 'none';
            error.style.display = 'none';
            loading.style.display = 'block';

            try {
                const response = await fetch('/api/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.style.display = 'none';

                if (data.error) {
                    error.textContent = data.error;
                    error.style.display = 'block';
                    return;
                }

                document.getElementById('cityName').textContent = data.name;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').src = data.icon_url;
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';
                document.getElementById('visibility').textContent = data.visibility + ' km';
                document.getElementById('clouds').textContent = data.clouds + '%';

                weatherInfo.style.display = 'block';
            } catch (err) {
                loading.style.display = 'none';
                error.textContent = 'Failed to fetch weather data. Please try again.';
                error.style.display = 'block';
            }
        }
    </script>
</body>
</html>"""


class WeatherHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path.startswith("/api/weather"):
            self.handle_weather_api()

        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

    def handle_weather_api(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()

        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        city = params.get("city", [""])[0].strip()

        if not city:
            result = {"error": "Please enter a city name."}
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if not API_KEY:
            result = self.get_demo_weather(city)
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        try:
            url = (
                "https://api.openweathermap.org/data/2.5/weather?"
                + urllib.parse.urlencode({"q": city, "appid": API_KEY, "units": "metric"})
            )

            req = urllib.request.Request(url)
            req.add_header("User-Agent", "WeatherApp/1.0")

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))

            visibility_km = round(data.get("visibility", 0) / 1000, 1)

            result = {
                "name": data["name"],
                "country": data["sys"].get("country", ""),
                "temperature": round(data["main"]["temp"], 1),
                "feels_like": round(data["main"]["feels_like"], 1),
                "description": data["weather"][0]["description"],
                "icon_url": "https://openweathermap.org/img/wn/"
                + data["weather"][0]["icon"]
                + "@2x.png",
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": round(data["wind"].get("speed", 0), 1),
                "visibility": visibility_km,
                "clouds": data["clouds"].get("all", 0),
            }

        except urllib.error.HTTPError as e:
            if e.code == 404:
                result = {"error": f"City '{city}' not found. Please check the spelling and try again."}
            elif e.code == 401:
                result = {"error": "Invalid API key. Please check your OPENWEATHERMAP_API_KEY."}
            else:
                result = {"error": f"API error (HTTP {e.code}). Please try again later."}
        except urllib.error.URLError:
            result = {"error": "Could not connect to weather service. Please check your internet connection."}
        except Exception as e:
            result = {"error": f"An unexpected error occurred: {str(e)}"}

        self.wfile.write(json.dumps(result).encode("utf-8"))

    def get_demo_weather(self, city):
        import hashlib
        import time

        city_hash = int(hashlib.md5(city.lower().encode()).hexdigest(), 16)

        demo_cities = {
            "london": {"name": "London", "country": "GB", "temp": 12.5, "feels": 10.2, "desc": "overcast clouds", "icon": "04d", "humidity": 78, "pressure": 1013, "wind": 5.2, "vis": 10.0, "clouds": 85},
            "new york": {"name": "New York", "country": "US", "temp": 18.3, "feels": 17.1, "desc": "clear sky", "icon": "01d", "humidity": 55, "pressure": 1018, "wind": 3.8, "vis": 10.0, "clouds": 5},
            "tokyo": {"name": "Tokyo", "country": "JP", "temp": 22.1, "feels": 21.5, "desc": "few clouds", "icon": "02d", "humidity": 65, "pressure": 1010, "wind": 4.1, "vis": 8.0, "clouds": 25},
            "paris": {"name": "Paris", "country": "FR", "temp": 15.8, "feels": 14.2, "desc": "light rain", "icon": "10d", "humidity": 82, "pressure": 1008, "wind": 6.3, "vis": 7.0, "clouds": 75},
            "sydney": {"name": "Sydney", "country": "AU", "temp": 24.6, "feels": 24.0, "desc": "scattered clouds", "icon": "03d", "humidity": 60, "pressure": 1015, "wind": 5.5, "vis": 10.0, "clouds": 40},
            "berlin": {"name": "Berlin", "country": "DE", "temp": 10.2, "feels": 8.5, "desc": "broken clouds", "icon": "04d", "humidity": 72, "pressure": 1011, "wind": 4.8, "vis": 9.0, "clouds": 65},
            "mumbai": {"name": "Mumbai", "country": "IN", "temp": 30.5, "feels": 34.2, "desc": "haze", "icon": "50d", "humidity": 75, "pressure": 1006, "wind": 3.2, "vis": 4.0, "clouds": 20},
            "moscow": {"name": "Moscow", "country": "RU", "temp": 5.1, "feels": 2.3, "desc": "snow", "icon": "13d", "humidity": 88, "pressure": 1005, "wind": 7.1, "vis": 3.0, "clouds": 95},
            "cairo": {"name": "Cairo", "country": "EG", "temp": 33.
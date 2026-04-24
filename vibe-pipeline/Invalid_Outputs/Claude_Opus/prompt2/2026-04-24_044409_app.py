import http.server
import socketserver
import urllib.request
import urllib.parse
import json
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
            width: 420px;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 25px;
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
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .city-name {
            font-size: 24px;
            color: #333;
            margin-bottom: 5px;
        }
        .country {
            font-size: 14px;
            color: #888;
            margin-bottom: 15px;
        }
        .weather-icon {
            font-size: 64px;
            margin: 10px 0;
        }
        .temperature {
            font-size: 52px;
            font-weight: bold;
            color: #333;
            margin: 5px 0;
        }
        .description {
            font-size: 18px;
            color: #666;
            text-transform: capitalize;
            margin-bottom: 20px;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .detail-item {
            background: #f8f9ff;
            padding: 15px;
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
            color: #333;
            font-weight: bold;
            margin-top: 5px;
        }
        .error {
            color: #e74c3c;
            font-size: 16px;
            margin-top: 15px;
            display: none;
        }
        .loading {
            display: none;
            margin: 20px 0;
            color: #667eea;
            font-size: 16px;
        }
        .api-notice {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #856404;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ Weather App</h1>
        <div class="api-notice" id="apiNotice">
            <strong>Note:</strong> Set the OPENWEATHERMAP_API_KEY environment variable with your free API key from
            <a href="https://openweathermap.org/api" target="_blank">openweathermap.org</a> before starting the server.
        </div>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." autofocus>
            <button onclick="getWeather()">Search</button>
        </div>
        <div class="loading" id="loading">⏳ Loading weather data...</div>
        <div class="error" id="error"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <div class="country" id="country"></div>
            <div class="weather-icon" id="weatherIcon"></div>
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
            </div>
        </div>
    </div>
    <script>
        const weatherEmojis = {
            '01d': '☀️', '01n': '🌙',
            '02d': '⛅', '02n': '☁️',
            '03d': '☁️', '03n': '☁️',
            '04d': '☁️', '04n': '☁️',
            '09d': '🌧️', '09n': '🌧️',
            '10d': '🌦️', '10n': '🌧️',
            '11d': '⛈️', '11n': '⛈️',
            '13d': '🌨️', '13n': '🌨️',
            '50d': '🌫️', '50n': '🌫️'
        };

        document.getElementById('cityInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') getWeather();
        });

        // Check if API key is configured
        fetch('/check_api_key')
            .then(r => r.json())
            .then(data => {
                if (!data.configured) {
                    document.getElementById('apiNotice').style.display = 'block';
                }
            });

        function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) return;

            const weatherInfo = document.getElementById('weatherInfo');
            const error = document.getElementById('error');
            const loading = document.getElementById('loading');

            weatherInfo.style.display = 'none';
            error.style.display = 'none';
            loading.style.display = 'block';

            fetch('/weather?city=' + encodeURIComponent(city))
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    if (data.error) {
                        error.textContent = data.error;
                        error.style.display = 'block';
                        return;
                    }

                    document.getElementById('cityName').textContent = data.name;
                    document.getElementById('country').textContent = data.sys.country;
                    document.getElementById('weatherIcon').textContent =
                        weatherEmojis[data.weather[0].icon] || '🌡️';
                    document.getElementById('temperature').textContent =
                        Math.round(data.main.temp) + '°C';
                    document.getElementById('description').textContent =
                        data.weather[0].description;
                    document.getElementById('feelsLike').textContent =
                        Math.round(data.main.feels_like) + '°C';
                    document.getElementById('humidity').textContent =
                        data.main.humidity + '%';
                    document.getElementById('wind').textContent =
                        data.wind.speed + ' m/s';
                    document.getElementById('pressure').textContent =
                        data.main.pressure + ' hPa';

                    weatherInfo.style.display = 'block';
                })
                .catch(err => {
                    loading.style.display = 'none';
                    error.textContent = 'Failed to fetch weather data. Please try again.';
                    error.style.display = 'block';
                });
        }
    </script>
</body>
</html>"""


class WeatherHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif self.path == '/check_api_key':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            result = {"configured": bool(API_KEY)}
            self.wfile.write(json.dumps(result).encode('utf-8'))

        elif self.path.startswith('/weather'):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            city = params.get('city', [''])[0]

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            if not city:
                result = {"error": "Please enter a city name."}
                self.wfile.write(json.dumps(result).encode('utf-8'))
                return

            if not API_KEY:
                result = self._get_demo_weather(city)
                self.wfile.write(json.dumps(result).encode('utf-8'))
                return

            try:
                encoded_city = urllib.parse.quote(city)
                url = (
                    f"https://api.openweathermap.org/data/2.5/weather"
                    f"?q={encoded_city}&appid={API_KEY}&units=metric"
                )
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'WeatherApp/1.0')

                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    self.wfile.write(json.dumps(data).encode('utf-8'))

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    result = {"error": f"City '{city}' not found. Please check the spelling."}
                elif e.code == 401:
                    result = {"error": "Invalid API key. Please check your OPENWEATHERMAP_API_KEY."}
                else:
                    result = {"error": f"API error: {e.code}"}
                self.wfile.write(json.dumps(result).encode('utf-8'))

            except Exception as e:
                result = {"error": f"Error fetching weather: {str(e)}"}
                self.wfile.write(json.dumps(result).encode('utf-8'))

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def _get_demo_weather(self, city):
        demo_cities = {
            "london": {"name": "London", "country": "GB", "temp": 15, "feels": 13, "humidity": 72, "wind": 5.1, "pressure": 1013, "desc": "overcast clouds", "icon": "04d"},
            "new york": {"name": "New York", "country": "US", "temp": 22, "feels": 21, "humidity": 55, "wind": 3.6, "pressure": 1015, "desc": "partly cloudy", "icon": "02d"},
            "tokyo": {"name": "Tokyo", "country": "JP", "temp": 28, "feels": 31, "humidity": 80, "wind": 2.1, "pressure": 1008, "desc": "light rain", "icon": "10d"},
            "paris": {"name": "Paris", "country": "FR", "temp": 18, "feels": 17, "humidity": 65, "wind": 4.2, "pressure": 1018, "desc": "clear sky", "icon": "01d"},
            "sydney": {"name": "Sydney", "country": "AU", "temp": 20, "feels": 19, "humidity": 60, "wind": 6.7, "pressure": 1022, "desc": "sunny", "icon": "01d"},
            "berlin": {"name": "Berlin", "country": "DE", "temp": 14, "feels": 12, "humidity": 68, "wind": 4.5, "pressure": 1010, "desc": "scattered clouds", "icon": "03d"},
            "mumbai": {"name": "Mumbai", "country": "IN", "temp": 32, "feels": 38, "humidity": 85, "wind": 3.1, "pressure": 1005, "desc": "haze", "icon": "50d"},
            "cairo": {"name": "Cairo", "country": "EG", "temp": 35, "feels": 33, "humidity": 25, "wind": 4.0, "pressure": 1012, "desc": "clear sky", "icon": "01d"},
            "moscow": {"name": "Moscow", "country": "RU", "temp": 8, "feels": 5, "humidity": 75, "wind": 5.5, "pressure": 1020, "desc": "light snow", "icon": "13d"},
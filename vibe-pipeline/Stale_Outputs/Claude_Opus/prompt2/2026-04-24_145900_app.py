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
            text-align: center;
            max-width: 500px;
            width: 90%;
        }
        h1 {
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
            border-radius: 30px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            border-color: #667eea;
        }
        button {
            padding: 12px 25px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 30px;
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
            animation: fadeIn 0.5s ease-in;
        }
        .weather-info.active {
            display: block;
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
            color: #888;
            font-size: 1em;
            margin-bottom: 15px;
        }
        .weather-icon {
            font-size: 4em;
            margin: 10px 0;
        }
        .temperature {
            font-size: 3.5em;
            font-weight: bold;
            color: #333;
        }
        .description {
            font-size: 1.2em;
            color: #666;
            text-transform: capitalize;
            margin: 10px 0 20px;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .detail-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 12px;
        }
        .detail-label {
            color: #888;
            font-size: 0.85em;
            margin-bottom: 5px;
        }
        .detail-value {
            color: #333;
            font-size: 1.2em;
            font-weight: 600;
        }
        .error {
            color: #e74c3c;
            font-size: 1.1em;
            margin-top: 20px;
            display: none;
        }
        .error.active {
            display: block;
        }
        .loading {
            display: none;
            margin-top: 20px;
            color: #666;
        }
        .loading.active {
            display: block;
        }
        .no-api-key {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            color: #856404;
            font-size: 0.9em;
            line-height: 1.5;
        }
        .no-api-key code {
            background: #f8f0d8;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.95em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤 Weather App</h1>
        <div id="apiWarning" class="no-api-key" style="display:none;">
            <strong>API Key Required!</strong><br>
            Set your OpenWeatherMap API key:<br>
            <code>export OPENWEATHERMAP_API_KEY=your_key_here</code><br>
            Get a free key at <a href="https://openweathermap.org/api" target="_blank">openweathermap.org</a>
        </div>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." autofocus>
            <button onclick="getWeather()">Search</button>
        </div>
        <div class="loading" id="loading">Loading weather data...</div>
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
                <div class="detail-item">
                    <div class="detail-label">Min Temp</div>
                    <div class="detail-value" id="tempMin"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Max Temp</div>
                    <div class="detail-value" id="tempMax"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const weatherEmojis = {
            'Clear': '☀️',
            'Clouds': '☁️',
            'Rain': '🌧️',
            'Drizzle': '🌦️',
            'Thunderstorm': '⛈️',
            'Snow': '❄️',
            'Mist': '🌫️',
            'Fog': '🌫️',
            'Haze': '🌫️',
            'Smoke': '🌫️',
            'Dust': '🌪️',
            'Sand': '🌪️',
            'Tornado': '🌪️'
        };

        document.getElementById('cityInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') getWeather();
        });

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) return;

            const weatherInfo = document.getElementById('weatherInfo');
            const error = document.getElementById('error');
            const loading = document.getElementById('loading');

            weatherInfo.classList.remove('active');
            error.classList.remove('active');
            loading.classList.add('active');

            try {
                const response = await fetch('/api/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.classList.remove('active');

                if (data.error) {
                    error.textContent = data.error;
                    error.classList.add('active');
                    return;
                }

                document.getElementById('cityName').textContent = data.name;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').textContent = weatherEmojis[data.main_weather] || '🌡️';
                document.getElementById('temperature').textContent = data.temp + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';
                document.getElementById('tempMin').textContent = data.temp_min + '°C';
                document.getElementById('tempMax').textContent = data.temp_max + '°C';

                weatherInfo.classList.add('active');
            } catch (err) {
                loading.classList.remove('active');
                error.textContent = 'Failed to fetch weather data. Please try again.';
                error.classList.add('active');
            }
        }

        // Check if API key is set
        fetch('/api/status').then(r => r.json()).then(data => {
            if (!data.api_key_set) {
                document.getElementById('apiWarning').style.display = 'block';
            }
        });
    </script>
</body>
</html>"""


class WeatherHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif self.path.startswith('/api/weather'):
            self.handle_weather_api()

        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            status = {"api_key_set": bool(API_KEY)}
            self.wfile.write(json.dumps(status).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def handle_weather_api(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        city = params.get('city', [''])[0]

        if not city:
            self.send_json_response({"error": "Please enter a city name."})
            return

        if not API_KEY:
            self.send_json_response({
                "error": "API key not configured. Set the OPENWEATHERMAP_API_KEY environment variable. "
                         "Get a free key at https://openweathermap.org/api"
            })
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

            weather_data = {
                "name": data.get("name", "Unknown"),
                "country": data.get("sys", {}).get("country", ""),
                "temp": round(data.get("main", {}).get("temp", 0), 1),
                "feels_like": round(data.get("main", {}).get("feels_like", 0), 1),
                "temp_min": round(data.get("main", {}).get("temp_min", 0), 1),
                "temp_max": round(data.get("main", {}).get("temp_max", 0), 1),
                "humidity": data.get("main", {}).get("humidity", 0),
                "pressure": data.get("main", {}).get("pressure", 0),
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "description": data.get("weather", [{}])[0].get("description", ""),
                "main_weather": data.get("weather", [{}])[0].get("main", ""),
            }

            self.send_json_response(weather_data)

        except urllib.error.HTTPError as e:
            if e.code == 404:
                self.send_json_response({"error": f"City '{city}' not found. Please check the spelling and try again."})
            elif e.code == 401:
                self.send_json_response({"error": "Invalid API key. Please check your OPENWEATHERMAP_API_KEY."})
            else:
                self.send_json_response({"error": f"API error (HTTP {e.code}). Please try again later."})

        except urllib.error.URLError:
            self.send_json_response({"error": "Network error. Please check your internet connection."})

        except Exception as e:
            self.send_json_response({"error": f"An unexpected error occurred: {str(e)}"})

    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    print("=" * 50)
    print("  🌤  Weather App")
    print("=" * 50)

    if not API_KEY:
        print()
        print("⚠️  WARNING: No API key found!")
        print("   Set your OpenWeatherMap API key:")
        print()
        print("   Linux/Mac:")
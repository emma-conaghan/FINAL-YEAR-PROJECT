import http.server
import json
import urllib.request
import urllib.parse
import urllib.error
import socketserver
import webbrowser
import threading
import os

PORT = 8080

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
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
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
            padding: 12px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102,126,234,0.4);
        }
        .weather-info {
            display: none;
            text-align: center;
        }
        .weather-info.active {
            display: block;
        }
        .city-name {
            font-size: 1.8em;
            color: #333;
            margin-bottom: 5px;
        }
        .country {
            color: #888;
            margin-bottom: 20px;
            font-size: 1.1em;
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
            font-size: 1.3em;
            color: #666;
            text-transform: capitalize;
            margin: 10px 0 20px 0;
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
            border-radius: 10px;
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
        .error-msg {
            display: none;
            color: #e74c3c;
            text-align: center;
            padding: 15px;
            background: #ffeaea;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .error-msg.active {
            display: block;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .loading.active {
            display: block;
        }
        .api-notice {
            text-align: center;
            color: #888;
            font-size: 0.8em;
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #eee;
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
        <div class="error-msg" id="errorMsg"></div>
        <div class="loading" id="loading">⏳ Fetching weather data...</div>
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
                    <div class="detail-value" id="windSpeed"></div>
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
        <div class="api-notice">
            Powered by OpenWeatherMap API
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
            'Ash': '🌋',
            'Squall': '💨',
            'Tornado': '🌪️'
        };

        document.getElementById('cityInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') getWeather();
        });

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) {
                showError('Please enter a city name.');
                return;
            }

            const weatherInfo = document.getElementById('weatherInfo');
            const errorMsg = document.getElementById('errorMsg');
            const loading = document.getElementById('loading');

            weatherInfo.classList.remove('active');
            errorMsg.classList.remove('active');
            loading.classList.add('active');

            try {
                const response = await fetch('/api/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.classList.remove('active');

                if (data.error) {
                    showError(data.error);
                    return;
                }

                document.getElementById('cityName').textContent = data.name;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').textContent = weatherEmojis[data.main_weather] || '🌡️';
                document.getElementById('temperature').textContent = data.temp + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('windSpeed').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';
                document.getElementById('tempMin').textContent = data.temp_min + '°C';
                document.getElementById('tempMax').textContent = data.temp_max + '°C';

                weatherInfo.classList.add('active');
            } catch (err) {
                loading.classList.remove('active');
                showError('Failed to fetch weather data. Please try again.');
            }
        }

        function showError(msg) {
            const errorMsg = document.getElementById('errorMsg');
            errorMsg.textContent = msg;
            errorMsg.classList.add('active');
            document.getElementById('weatherInfo').classList.remove('active');
        }
    </script>
</body>
</html>"""


def fetch_weather(city):
    if not API_KEY:
        return {"error": "No API key set. Please set the OPENWEATHERMAP_API_KEY environment variable. "
                         "Get a free key at https://openweathermap.org/api"}

    encoded_city = urllib.parse.quote(city)
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?q={encoded_city}&appid={API_KEY}&units=metric")

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'WeatherApp/1.0')
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))

        result = {
            "name": data.get("name", ""),
            "country": data.get("sys", {}).get("country", ""),
            "temp": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "temp_min": round(data["main"]["temp_min"], 1),
            "temp_max": round(data["main"]["temp_max"], 1),
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": round(data.get("wind", {}).get("speed", 0), 1),
            "description": data["weather"][0]["description"] if data.get("weather") else "",
            "main_weather": data["weather"][0]["main"] if data.get("weather") else "",
        }
        return result

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"City '{city}' not found. Please check the spelling and try again."}
        elif e.code == 401:
            return {"error": "Invalid API key. Please check your OPENWEATHERMAP_API_KEY."}
        else:
            return {"error": f"API error (HTTP {e.code}). Please try again later."}
    except urllib.error.URLError:
        return {"error": "Could not connect to the weather service. Check your internet connection."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


class WeatherHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif self.path.startswith('/api/weather'):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            city = params.get('city', [''])[0]

            if not city:
                result = {"error": "Please provide a city name."}
            else:
                result = fetch_weather(city)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def log_message(self, format_str, *args):
        print(f"[{self.log_date_time_string()}] {args[0] if args else ''}")


def main():
    if not API_KEY:
        print("=" * 60)
        print("WARNING: No OpenWeatherMap API key detected!")
        print("")
        print("To use this app, you need a free API key from:")
        print("  https://openweathermap.org/api")
        print("")
        print("Then set it as an environment variable:")
        print("  export OPENWEATHERMAP_API_KEY=your_key_here  (Linux/Mac)")
        print("  set OPENWEATHERMAP_API_KEY=your_key_here     (Windows)")
        print("")
        print("Or run with:")
        print("  OPENWEATHERMAP_API_KEY=your_key python app.py")
        print("=" * 60)
    else:
        print("✓ API key detected.")

    with socketserver.TCPServer(("", PORT), WeatherHandler) as httpd:
        print(f"\n🌤  Weather App running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.\n")

        def open_browser():
            webbrowser.open(f"http://localhost:{PORT}")

        threading.Timer(1.0, open_browser).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()


if __name__ == "__main__":
    main()
import http.server
import socketserver
import urllib.request
import urllib.parse
import json
import os

PORT = 8000

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
            margin-bottom: 20px;
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
            border-radius: 10px;
        }
        .detail-label {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 5px;
        }
        .detail-value {
            font-size: 1.2em;
            color: #333;
            font-weight: 600;
        }
        .error {
            display: none;
            text-align: center;
            color: #e74c3c;
            font-size: 1.1em;
            padding: 20px;
            background: #ffeaea;
            border-radius: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #ddd;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-left: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .api-note {
            text-align: center;
            font-size: 0.8em;
            color: #aaa;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ Weather App</h1>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." onkeypress="if(event.key==='Enter')getWeather()">
            <button onclick="getWeather()">Search</button>
        </div>
        <div class="loading" id="loading">Fetching weather data</div>
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
        <div class="api-note">Powered by Open-Meteo (no API key required)</div>
    </div>
    <script>
        const weatherIcons = {
            0: '☀️', 1: '🌤️', 2: '⛅', 3: '☁️',
            45: '🌫️', 48: '🌫️',
            51: '🌦️', 53: '🌦️', 55: '🌧️',
            56: '🌨️', 57: '🌨️',
            61: '🌧️', 63: '🌧️', 65: '🌧️',
            66: '🌨️', 67: '🌨️',
            71: '🌨️', 73: '🌨️', 75: '❄️',
            77: '❄️',
            80: '🌦️', 81: '🌧️', 82: '⛈️',
            85: '🌨️', 86: '🌨️',
            95: '⛈️', 96: '⛈️', 99: '⛈️'
        };

        const weatherDescriptions = {
            0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Foggy', 48: 'Depositing rime fog',
            51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
            56: 'Light freezing drizzle', 57: 'Dense freezing drizzle',
            61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
            66: 'Light freezing rain', 67: 'Heavy freezing rain',
            71: 'Slight snowfall', 73: 'Moderate snowfall', 75: 'Heavy snowfall',
            77: 'Snow grains',
            80: 'Slight rain showers', 81: 'Moderate rain showers', 82: 'Violent rain showers',
            85: 'Slight snow showers', 86: 'Heavy snow showers',
            95: 'Thunderstorm', 96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'
        };

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) {
                showError('Please enter a city name.');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('weatherInfo').style.display = 'none';
            document.getElementById('error').style.display = 'none';

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';

                if (data.error) {
                    showError(data.error);
                    return;
                }

                const code = data.weather_code;
                document.getElementById('cityName').textContent = data.city;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').textContent = weatherIcons[code] || '🌡️';
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('description').textContent = weatherDescriptions[code] || 'Unknown';
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' km/h';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';
                document.getElementById('weatherInfo').style.display = 'block';
            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                showError('Failed to fetch weather data. Please try again.');
            }
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            document.getElementById('weatherInfo').style.display = 'none';
        }
    </script>
</body>
</html>"""


class WeatherHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path.startswith('/weather?'):
            self.handle_weather()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_weather(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        city = params.get('city', [''])[0]

        if not city:
            self.send_json({'error': 'City name is required.'})
            return

        try:
            geo_url = (
                'https://geocoding-api.open-meteo.com/v1/search?name='
                + urllib.parse.quote(city)
                + '&count=1&language=en&format=json'
            )
            geo_req = urllib.request.Request(geo_url, headers={'User-Agent': 'WeatherApp/1.0'})
            with urllib.request.urlopen(geo_req, timeout=10) as response:
                geo_data = json.loads(response.read().decode('utf-8'))

            if 'results' not in geo_data or len(geo_data['results']) == 0:
                self.send_json({'error': f'City "{city}" not found. Please check the spelling.'})
                return

            location = geo_data['results'][0]
            lat = location['latitude']
            lon = location['longitude']
            city_name = location.get('name', city)
            country = location.get('country', '')
            admin1 = location.get('admin1', '')

            country_display = f"{admin1}, {country}" if admin1 else country

            weather_url = (
                f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
                f'&current=temperature_2m,relative_humidity_2m,apparent_temperature,'
                f'surface_pressure,wind_speed_10m,weather_code'
            )
            weather_req = urllib.request.Request(weather_url, headers={'User-Agent': 'WeatherApp/1.0'})
            with urllib.request.urlopen(weather_req, timeout=10) as response:
                weather_data = json.loads(response.read().decode('utf-8'))

            current = weather_data['current']

            result = {
                'city': city_name,
                'country': country_display,
                'temperature': current.get('temperature_2m', 'N/A'),
                'feels_like': current.get('apparent_temperature', 'N/A'),
                'humidity': current.get('relative_humidity_2m', 'N/A'),
                'wind_speed': current.get('wind_speed_10m', 'N/A'),
                'pressure': current.get('surface_pressure', 'N/A'),
                'weather_code': current.get('weather_code', 0),
            }

            self.send_json(result)

        except urllib.error.URLError as e:
            self.send_json({'error': f'Network error: {str(e)}'})
        except Exception as e:
            self.send_json({'error': f'An error occurred: {str(e)}'})

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    print(f"🌤️  Weather App starting...")
    print(f"📡 Using Open-Meteo API (no API key required)")
    print(f"🌐 Open your browser to: http
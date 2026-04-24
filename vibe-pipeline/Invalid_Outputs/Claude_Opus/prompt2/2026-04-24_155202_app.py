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
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
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
            border-radius: 50px;
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
            border-radius: 50px;
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
            margin-bottom: 20px;
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
            font-size: 1.1em;
            font-weight: 600;
        }
        .error {
            color: #e74c3c;
            margin-top: 15px;
            display: none;
            font-weight: 500;
        }
        .loading {
            display: none;
            color: #667eea;
            margin-top: 15px;
        }
        .api-notice {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 0.85em;
            color: #856404;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ Weather App</h1>
        <p class="subtitle">Enter a city name to get current weather</p>
        
        <div id="apiNotice" class="api-notice">
            <strong>Note:</strong> No API key detected. Set the environment variable 
            <code>OPENWEATHERMAP_API_KEY</code> with your free key from 
            <a href="https://openweathermap.org/api" target="_blank">openweathermap.org</a>.
            The app will use demo data without a valid key.
        </div>

        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." 
                   onkeypress="if(event.key==='Enter')getWeather()">
            <button onclick="getWeather()">Search</button>
        </div>

        <div class="loading" id="loading">⏳ Fetching weather data...</div>
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
            'Clear': '☀️',
            'Clouds': '☁️',
            'Rain': '🌧️',
            'Drizzle': '🌦️',
            'Thunderstorm': '⛈️',
            'Snow': '❄️',
            'Mist': '🌫️',
            'Haze': '🌫️',
            'Fog': '🌫️',
            'Smoke': '🌫️',
            'Dust': '🌪️',
            'Sand': '🌪️',
            'Tornado': '🌪️'
        };

        async function checkApiKey() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                if (!data.has_api_key) {
                    document.getElementById('apiNotice').style.display = 'block';
                }
            } catch(e) {}
        }

        checkApiKey();

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) {
                showError('Please enter a city name.');
                return;
            }

            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const weatherInfo = document.getElementById('weatherInfo');

            loading.style.display = 'block';
            error.style.display = 'none';
            weatherInfo.style.display = 'none';

            try {
                const response = await fetch('/api/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.style.display = 'none';

                if (data.error) {
                    showError(data.error);
                    return;
                }

                document.getElementById('cityName').textContent = data.city;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').textContent = 
                    weatherEmojis[data.main] || '🌡️';
                document.getElementById('temperature').textContent = 
                    Math.round(data.temp) + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = 
                    Math.round(data.feels_like) + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';

                weatherInfo.style.display = 'block';
            } catch (err) {
                loading.style.display = 'none';
                showError('Failed to fetch weather data. Please try again.');
            }
        }

        function showError(message) {
            const error = document.getElementById('error');
            error.textContent = message;
            error.style.display = 'block';
        }
    </script>
</body>
</html>"""

DEMO_DATA = {
    "london": {"city": "London", "country": "United Kingdom", "temp": 15.2, "feels_like": 13.8, "humidity": 72, "pressure": 1013, "wind_speed": 4.1, "description": "overcast clouds", "main": "Clouds"},
    "new york": {"city": "New York", "country": "United States", "temp": 22.5, "feels_like": 21.0, "humidity": 58, "pressure": 1018, "wind_speed": 3.6, "description": "clear sky", "main": "Clear"},
    "tokyo": {"city": "Tokyo", "country": "Japan", "temp": 28.3, "feels_like": 30.1, "humidity": 65, "pressure": 1008, "wind_speed": 2.5, "description": "light rain", "main": "Rain"},
    "paris": {"city": "Paris", "country": "France", "temp": 18.7, "feels_like": 17.2, "humidity": 60, "pressure": 1015, "wind_speed": 3.2, "description": "few clouds", "main": "Clouds"},
    "sydney": {"city": "Sydney", "country": "Australia", "temp": 20.1, "feels_like": 19.5, "humidity": 55, "pressure": 1020, "wind_speed": 5.0, "description": "sunny", "main": "Clear"},
    "berlin": {"city": "Berlin", "country": "Germany", "temp": 14.0, "feels_like": 12.5, "humidity": 68, "pressure": 1012, "wind_speed": 4.5, "description": "scattered clouds", "main": "Clouds"},
    "mumbai": {"city": "Mumbai", "country": "India", "temp": 32.0, "feels_like": 36.0, "humidity": 80, "pressure": 1005, "wind_speed": 3.8, "description": "haze", "main": "Haze"},
    "cairo": {"city": "Cairo", "country": "Egypt", "temp": 35.5, "feels_like": 34.0, "humidity": 25, "pressure": 1010, "wind_speed": 2.0, "description": "clear sky", "main": "Clear"},
    "moscow": {"city": "Moscow", "country": "Russia", "temp": 8.0, "feels_like": 5.5, "humidity": 75, "pressure": 1016, "wind_speed": 6.0, "description": "light snow", "main": "Snow"},
    "toronto": {"city": "Toronto", "country": "Canada", "temp": 17.3, "feels_like": 16.0, "humidity": 62, "pressure": 1014, "wind_speed": 3.9, "description": "partly cloudy", "main": "Clouds"},
    "san francisco": {"city": "San Francisco", "country": "United States", "temp": 16.5, "feels_like": 15.0, "humidity": 78, "pressure": 1017, "wind_speed": 5.5, "description": "fog", "main": "Fog"},
    "beijing": {"city": "Beijing", "country": "China", "temp": 26.0, "feels_like": 25.0, "humidity": 45, "pressure": 1011, "wind_speed": 2.8, "description": "haze", "main": "Haze"},
    "rome": {"city": "Rome", "country": "Italy", "temp": 24.0, "feels_like": 23.5, "humidity": 50, "pressure": 1016, "wind_speed": 2.3, "description": "clear sky", "main": "Clear"},
    "dubai": {"city": "Dubai", "country": "UAE", "temp": 40.0, "feels_like": 42.0, "humidity": 30, "pressure": 1006, "wind_speed": 3.0, "description": "clear sky", "main": "Clear"},
    "singapore": {"city": "Singapore", "country": "Singapore", "temp": 30.0, "feels_like": 34.0, "humidity": 85, "pressure": 1009, "wind_speed": 2.0, "description": "thunderstorm", "main": "Thunderstorm"},
}


def fetch_weather_from_api(city):
    """Fetch weather data from OpenWeatherMap API."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = urllib.parse.urlencode({
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    })
    url = f"{base_url}?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "WeatherApp/1.0")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        if data.get("cod") != 200:
            return {"error": data.get("message", "City not found.")}

        result = {
            "city": data["name"],
            "country": data.get("sys", {}).get("country", ""),
            "temp": data["main"]["
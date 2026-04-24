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
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .weather-info {
            display: none;
        }
        .weather-info.active {
            display: block;
        }
        .city-name {
            font-size: 24px;
            color: #333;
            margin-bottom: 5px;
        }
        .country {
            font-size: 14px;
            color: #888;
            margin-bottom: 20px;
        }
        .weather-icon {
            font-size: 64px;
            margin-bottom: 10px;
        }
        .temperature {
            font-size: 52px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .description {
            font-size: 18px;
            color: #666;
            text-transform: capitalize;
            margin-bottom: 25px;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .detail-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
        }
        .detail-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .detail-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .error {
            color: #e74c3c;
            font-size: 16px;
            margin-top: 15px;
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
        .api-note {
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 10px;
            font-size: 13px;
            color: #856404;
            display: none;
        }
        .api-note.active {
            display: block;
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
        <div class="loading" id="loading">Loading...</div>
        <div class="error" id="error"></div>
        <div class="api-note" id="apiNote"></div>
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

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) {
                showError('Please enter a city name.');
                return;
            }

            const weatherInfo = document.getElementById('weatherInfo');
            const error = document.getElementById('error');
            const loading = document.getElementById('loading');
            const apiNote = document.getElementById('apiNote');

            weatherInfo.classList.remove('active');
            error.classList.remove('active');
            apiNote.classList.remove('active');
            loading.classList.add('active');

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.classList.remove('active');

                if (data.error) {
                    showError(data.error);
                    if (data.note) {
                        apiNote.textContent = data.note;
                        apiNote.classList.add('active');
                    }
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

                weatherInfo.classList.add('active');
            } catch (err) {
                loading.classList.remove('active');
                showError('Failed to fetch weather data. Please try again.');
            }
        }

        function showError(msg) {
            const error = document.getElementById('error');
            error.textContent = msg;
            error.classList.add('active');
        }

        document.getElementById('cityInput').focus();
    </script>
</body>
</html>"""

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

FALLBACK_DATA = {
    "london": {"name": "London", "country": "United Kingdom", "temp": 15, "feels_like": 13, "humidity": 72, "pressure": 1013, "wind_speed": 5.1, "description": "overcast clouds", "main_weather": "Clouds"},
    "new york": {"name": "New York", "country": "United States", "temp": 22, "feels_like": 21, "humidity": 65, "pressure": 1015, "wind_speed": 3.6, "description": "partly cloudy", "main_weather": "Clouds"},
    "tokyo": {"name": "Tokyo", "country": "Japan", "temp": 28, "feels_like": 30, "humidity": 78, "pressure": 1008, "wind_speed": 2.1, "description": "light rain", "main_weather": "Rain"},
    "paris": {"name": "Paris", "country": "France", "temp": 18, "feels_like": 17, "humidity": 68, "pressure": 1020, "wind_speed": 4.2, "description": "clear sky", "main_weather": "Clear"},
    "sydney": {"name": "Sydney", "country": "Australia", "temp": 20, "feels_like": 19, "humidity": 55, "pressure": 1022, "wind_speed": 6.7, "description": "sunny", "main_weather": "Clear"},
    "berlin": {"name": "Berlin", "country": "Germany", "temp": 14, "feels_like": 12, "humidity": 74, "pressure": 1018, "wind_speed": 3.8, "description": "scattered clouds", "main_weather": "Clouds"},
    "mumbai": {"name": "Mumbai", "country": "India", "temp": 32, "feels_like": 38, "humidity": 85, "pressure": 1006, "wind_speed": 4.5, "description": "haze", "main_weather": "Haze"},
    "cairo": {"name": "Cairo", "country": "Egypt", "temp": 35, "feels_like": 33, "humidity": 25, "pressure": 1012, "wind_speed": 3.2, "description": "clear sky", "main_weather": "Clear"},
    "moscow": {"name": "Moscow", "country": "Russia", "temp": 8, "feels_like": 5, "humidity": 80, "pressure": 1010, "wind_speed": 5.5, "description": "light snow", "main_weather": "Snow"},
    "toronto": {"name": "Toronto", "country": "Canada", "temp": 16, "feels_like": 14, "humidity": 60, "pressure": 1017, "wind_speed": 4.0, "description": "few clouds", "main_weather": "Clouds"},
    "dubai": {"name": "Dubai", "country": "United Arab Emirates", "temp": 40, "feels_like": 43, "humidity": 30, "pressure": 1005, "wind_speed": 3.0, "description": "clear sky", "main_weather": "Clear"},
    "beijing": {"name": "Beijing", "country": "China", "temp": 26, "feels_like": 27, "humidity": 50, "pressure": 1014, "wind_speed": 2.5, "description": "haze", "main_weather": "Haze"},
    "rome": {"name": "Rome", "country": "Italy", "temp": 25, "feels_like": 24, "humidity": 45, "pressure": 1019, "wind_speed": 3.1, "description": "clear sky", "main_weather": "Clear"},
    "los angeles": {"name": "Los Angeles", "country": "United States", "temp": 27, "feels_like": 26, "humidity": 40, "pressure": 1016, "wind_speed": 2.8, "description": "sunny", "main_weather": "Clear"},
    "chicago": {"name": "Chicago", "country": "United States", "temp": 19, "feels_like": 17, "humidity": 58, "pressure": 1014, "wind_speed": 6.2, "description": "partly cloudy", "main_weather": "Clouds"},
    "singapore": {"name": "Singapore", "country": "Singapore", "temp": 31, "feels_like": 35, "humidity": 82, "pressure": 1009, "wind_speed": 2.0, "description": "thunderstorm", "main_weather": "Thunderstorm"},
    "amsterdam": {"name": "Amsterdam", "country": "Netherlands", "temp": 13, "feels_like": 11, "humidity": 76, "pressure": 1015, "wind_speed": 5.8, "description": "drizzle", "main_weather": "Drizzle"},
    "san francisco": {"name": "San Francisco", "country": "United States", "temp": 17, "feels_like": 15, "humidity": 70, "pressure": 1018, "wind_speed": 5.0, "description": "fog", "main_weather": "Fog"},
    "seoul": {"name": "Seoul", "country": "South Korea", "temp": 24, "feels_like": 25, "humidity": 62, "pressure": 1011, "wind_speed": 2.3, "description": "clear sky", "main_weather": "Clear"},
    "bangkok": {"name": "Bangkok", "country": "Thailand", "temp": 33, "feels_like": 38, "humidity": 75, "pressure": 1007, "wind_speed": 1.8, "description": "scattered clouds", "main_weather": "Clouds"},
}


def fetch_weather_from_api(city):
    if not API_KEY:
        return None, "No API key configured"

    encoded_city = urllib.parse.quote(city)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_city}&appid={API_KEY}&units=metric"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WeatherApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
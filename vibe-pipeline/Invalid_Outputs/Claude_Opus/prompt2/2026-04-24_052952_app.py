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
            border-radius: 10px;
            padding: 15px;
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
        .error.active {
            display: block;
        }
        .loading {
            display: none;
            margin: 20px 0;
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
            '01d': '☀️', '01n': '🌙',
            '02d': '⛅', '02n': '☁️',
            '03d': '☁️', '03n': '☁️',
            '04d': '☁️', '04n': '☁️',
            '09d': '🌧️', '09n': '🌧️',
            '10d': '🌦️', '10n': '🌧️',
            '11d': '⛈️', '11n': '⛈️',
            '13d': '❄️', '13n': '❄️',
            '50d': '🌫️', '50n': '🌫️'
        };

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) return;

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
                    error.textContent = data.error;
                    error.classList.add('active');
                    if (data.note) {
                        apiNote.textContent = data.note;
                        apiNote.classList.add('active');
                    }
                    return;
                }

                document.getElementById('cityName').textContent = data.name;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').textContent = weatherEmojis[data.icon] || '🌡️';
                document.getElementById('temperature').textContent = data.temp + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';

                weatherInfo.classList.add('active');
            } catch (err) {
                loading.classList.remove('active');
                error.textContent = 'Failed to fetch weather data. Please try again.';
                error.classList.add('active');
            }
        }
    </script>
</body>
</html>"""

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

FALLBACK_DATA = {
    "london": {"name": "London", "country": "United Kingdom", "temp": 15, "feels_like": 13, "humidity": 72, "wind": 5.1, "pressure": 1013, "description": "overcast clouds", "icon": "04d"},
    "new york": {"name": "New York", "country": "United States", "temp": 22, "feels_like": 21, "humidity": 55, "wind": 3.6, "pressure": 1015, "description": "partly cloudy", "icon": "02d"},
    "tokyo": {"name": "Tokyo", "country": "Japan", "temp": 28, "feels_like": 30, "humidity": 68, "wind": 4.2, "pressure": 1008, "description": "light rain", "icon": "10d"},
    "paris": {"name": "Paris", "country": "France", "temp": 18, "feels_like": 17, "humidity": 65, "wind": 3.8, "pressure": 1018, "description": "clear sky", "icon": "01d"},
    "sydney": {"name": "Sydney", "country": "Australia", "temp": 20, "feels_like": 19, "humidity": 60, "wind": 6.2, "pressure": 1020, "description": "sunny", "icon": "01d"},
    "berlin": {"name": "Berlin", "country": "Germany", "temp": 14, "feels_like": 12, "humidity": 70, "wind": 4.5, "pressure": 1010, "description": "scattered clouds", "icon": "03d"},
    "moscow": {"name": "Moscow", "country": "Russia", "temp": 8, "feels_like": 5, "humidity": 80, "wind": 5.5, "pressure": 1005, "description": "overcast clouds", "icon": "04d"},
    "dubai": {"name": "Dubai", "country": "UAE", "temp": 38, "feels_like": 42, "humidity": 45, "wind": 3.0, "pressure": 1002, "description": "clear sky", "icon": "01d"},
    "mumbai": {"name": "Mumbai", "country": "India", "temp": 32, "feels_like": 36, "humidity": 78, "wind": 4.8, "pressure": 1006, "description": "haze", "icon": "50d"},
    "beijing": {"name": "Beijing", "country": "China", "temp": 25, "feels_like": 24, "humidity": 50, "wind": 3.2, "pressure": 1012, "description": "few clouds", "icon": "02d"},
    "cairo": {"name": "Cairo", "country": "Egypt", "temp": 35, "feels_like": 33, "humidity": 25, "wind": 4.0, "pressure": 1011, "description": "clear sky", "icon": "01d"},
    "rome": {"name": "Rome", "country": "Italy", "temp": 24, "feels_like": 23, "humidity": 55, "wind": 2.8, "pressure": 1016, "description": "sunny", "icon": "01d"},
    "toronto": {"name": "Toronto", "country": "Canada", "temp": 18, "feels_like": 16, "humidity": 62, "wind": 4.1, "pressure": 1014, "description": "partly cloudy", "icon": "02d"},
    "singapore": {"name": "Singapore", "country": "Singapore", "temp": 30, "feels_like": 34, "humidity": 82, "wind": 2.5, "pressure": 1009, "description": "thunderstorm", "icon": "11d"},
    "los angeles": {"name": "Los Angeles", "country": "United States", "temp": 26, "feels_like": 25, "humidity": 40, "wind": 3.5, "pressure": 1017, "description": "clear sky", "icon": "01d"},
    "chicago": {"name": "Chicago", "country": "United States", "temp": 19, "feels_like": 17, "humidity": 58, "wind": 6.0, "pressure": 1013, "description": "windy", "icon": "03d"},
    "san francisco": {"name": "San Francisco", "country": "United States", "temp": 17, "feels_like": 15, "humidity": 75, "wind": 5.8, "pressure": 1016, "description": "foggy", "icon": "50d"},
    "amsterdam": {"name": "Amsterdam", "country": "Netherlands", "temp": 13, "feels_like": 11, "humidity": 78, "wind": 5.0, "pressure": 1011, "description": "light rain", "icon": "10d"},
    "madrid": {"name": "Madrid", "country": "Spain", "temp": 27, "feels_like": 26, "humidity": 35, "wind": 3.3, "pressure": 1019, "description": "clear sky", "icon": "01d"},
    "seoul": {"name": "Seoul", "country": "South Korea", "temp": 23, "feels_like": 22, "humidity": 60, "wind": 3.7, "pressure": 1014, "description": "scattered clouds", "icon": "03d"},
}


def fetch_weather_from_api(city):
    if not API_KEY:
        return None

    try:
        encoded_city = urllib.parse.quote(city)
        url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_city}&appid={API_KEY}&units=metric"
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'WeatherApp/1.0')

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        result = {
            "name": data["name"],
            "country": data["sys"].get("country", ""),
            "temp": round(data["main"]["temp"]),
            "feels_like": round(data["main"]["feels_like"]),
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"],
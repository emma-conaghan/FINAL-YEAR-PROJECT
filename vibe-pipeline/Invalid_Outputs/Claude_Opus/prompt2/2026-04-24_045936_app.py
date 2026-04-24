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
            transform: scale(1.05);
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
            margin-bottom: 10px;
        }
        .weather-icon {
            width: 100px;
            height: 100px;
        }
        .temperature {
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .description {
            font-size: 1.2em;
            color: #666;
            text-transform: capitalize;
            margin-bottom: 20px;
        }
        .details {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .detail-item {
            text-align: center;
        }
        .detail-label {
            font-size: 0.85em;
            color: #999;
            margin-bottom: 5px;
        }
        .detail-value {
            font-size: 1.1em;
            color: #333;
            font-weight: 600;
        }
        .error {
            color: #e74c3c;
            font-size: 1.1em;
            margin-top: 15px;
            display: none;
        }
        .loading {
            display: none;
            margin: 20px 0;
            color: #667eea;
            font-size: 1.1em;
        }
        .api-notice {
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 10px;
            color: #856404;
            font-size: 0.85em;
            display: none;
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
        <div class="loading" id="loading">Loading weather data...</div>
        <div class="error" id="error"></div>
        <div class="api-notice" id="apiNotice"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <img class="weather-icon" id="weatherIcon" src="" alt="weather icon">
            <div class="temperature" id="temperature"></div>
            <div class="description" id="description"></div>
            <div class="details">
                <div class="detail-item">
                    <div class="detail-label">Humidity</div>
                    <div class="detail-value" id="humidity"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Wind Speed</div>
                    <div class="detail-value" id="wind"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Feels Like</div>
                    <div class="detail-value" id="feelsLike"></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').style.display = 'none';
            document.getElementById('weatherInfo').style.display = 'none';
            document.getElementById('apiNotice').style.display = 'none';

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';

                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                    if (data.notice) {
                        document.getElementById('apiNotice').textContent = data.notice;
                        document.getElementById('apiNotice').style.display = 'block';
                    }
                    return;
                }

                document.getElementById('cityName').textContent = data.city + ', ' + data.country;
                document.getElementById('weatherIcon').src = data.icon;
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('weatherInfo').style.display = 'block';
            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').textContent = 'Failed to fetch weather data. Please try again.';
                document.getElementById('error').style.display = 'block';
            }
        }
    </script>
</body>
</html>"""

DEMO_WEATHER_DATA = {
    "london": {"city": "London", "country": "GB", "temperature": 12, "feels_like": 10, "description": "overcast clouds", "humidity": 78, "wind_speed": 5.2, "icon": "https://openweathermap.org/img/wn/04d@2x.png"},
    "new york": {"city": "New York", "country": "US", "temperature": 18, "feels_like": 16, "description": "clear sky", "humidity": 55, "wind_speed": 3.1, "icon": "https://openweathermap.org/img/wn/01d@2x.png"},
    "paris": {"city": "Paris", "country": "FR", "temperature": 15, "feels_like": 13, "description": "few clouds", "humidity": 65, "wind_speed": 4.0, "icon": "https://openweathermap.org/img/wn/02d@2x.png"},
    "tokyo": {"city": "Tokyo", "country": "JP", "temperature": 22, "feels_like": 21, "description": "scattered clouds", "humidity": 70, "wind_speed": 2.5, "icon": "https://openweathermap.org/img/wn/03d@2x.png"},
    "sydney": {"city": "Sydney", "country": "AU", "temperature": 20, "feels_like": 19, "description": "light rain", "humidity": 80, "wind_speed": 6.0, "icon": "https://openweathermap.org/img/wn/10d@2x.png"},
    "berlin": {"city": "Berlin", "country": "DE", "temperature": 10, "feels_like": 8, "description": "broken clouds", "humidity": 72, "wind_speed": 4.5, "icon": "https://openweathermap.org/img/wn/04d@2x.png"},
    "moscow": {"city": "Moscow", "country": "RU", "temperature": 5, "feels_like": 2, "description": "snow", "humidity": 85, "wind_speed": 3.8, "icon": "https://openweathermap.org/img/wn/13d@2x.png"},
    "dubai": {"city": "Dubai", "country": "AE", "temperature": 35, "feels_like": 38, "description": "clear sky", "humidity": 40, "wind_speed": 2.0, "icon": "https://openweathermap.org/img/wn/01d@2x.png"},
    "mumbai": {"city": "Mumbai", "country": "IN", "temperature": 30, "feels_like": 33, "description": "haze", "humidity": 75, "wind_speed": 3.5, "icon": "https://openweathermap.org/img/wn/50d@2x.png"},
    "beijing": {"city": "Beijing", "country": "CN", "temperature": 16, "feels_like": 14, "description": "partly cloudy", "humidity": 50, "wind_speed": 4.2, "icon": "https://openweathermap.org/img/wn/02d@2x.png"},
    "rome": {"city": "Rome", "country": "IT", "temperature": 19, "feels_like": 18, "description": "sunny", "humidity": 55, "wind_speed": 2.8, "icon": "https://openweathermap.org/img/wn/01d@2x.png"},
    "toronto": {"city": "Toronto", "country": "CA", "temperature": 14, "feels_like": 12, "description": "cloudy", "humidity": 60, "wind_speed": 5.0, "icon": "https://openweathermap.org/img/wn/04d@2x.png"},
    "los angeles": {"city": "Los Angeles", "country": "US", "temperature": 24, "feels_like": 23, "description": "clear sky", "humidity": 45, "wind_speed": 2.3, "icon": "https://openweathermap.org/img/wn/01d@2x.png"},
    "chicago": {"city": "Chicago", "country": "US", "temperature": 11, "feels_like": 8, "description": "windy", "humidity": 58, "wind_speed": 7.5, "icon": "https://openweathermap.org/img/wn/03d@2x.png"},
    "san francisco": {"city": "San Francisco", "country": "US", "temperature": 16, "feels_like": 14, "description": "fog", "humidity": 82, "wind_speed": 4.1, "icon": "https://openweathermap.org/img/wn/50d@2x.png"},
    "singapore": {"city": "Singapore", "country": "SG", "temperature": 31, "feels_like": 34, "description": "thunderstorm", "humidity": 88, "wind_speed": 3.0, "icon": "https://openweathermap.org/img/wn/11d@2x.png"},
    "amsterdam": {"city": "Amsterdam", "country": "NL", "temperature": 11, "feels_like": 9, "description": "light rain", "humidity": 76, "wind_speed": 5.5, "icon": "https://openweathermap.org/img/wn/10d@2x.png"},
    "cairo": {"city": "Cairo", "country": "EG", "temperature": 28, "feels_like": 27, "description": "clear sky", "humidity": 35, "wind_speed": 3.2, "icon": "https://openweathermap.org/img/wn/01d@2x.png"},
    "seoul": {"city": "Seoul", "country": "KR", "temperature": 17, "feels_like": 15, "description": "mist", "humidity": 68, "wind_speed": 2.9, "icon": "https://openweathermap.org/img/wn/50d@2x.png"},
    "madrid": {"city": "Madrid", "country": "ES", "temperature": 21, "feels_like": 20, "description": "clear sky", "humidity": 42, "wind_speed": 3.6, "icon": "https://openweathermap.org/img/wn/01d@2x.png"},
}


def fetch_weather_from_api(city):
    """Fetch weather from OpenWeatherMap API."""
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = urllib.parse.urlencode({
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    })
    url = f"{base_url}?{params}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        result = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": round(data["main"]["temp"]),
            "feels_like": round(data["main"]["feels_like"]),
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "icon": f
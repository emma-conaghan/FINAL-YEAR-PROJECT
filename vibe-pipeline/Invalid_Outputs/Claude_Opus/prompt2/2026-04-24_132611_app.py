import http.server
import json
import urllib.request
import urllib.parse
import socketserver
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
            padding: 15px 20px;
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
            padding: 15px 25px;
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
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            border-radius: 15px;
            margin-top: 20px;
        }
        .weather-info.active {
            display: block;
        }
        .city-name {
            font-size: 1.8em;
            color: #333;
            margin-bottom: 10px;
        }
        .temperature {
            font-size: 3.5em;
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
        .weather-icon {
            font-size: 4em;
            margin: 10px 0;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .detail-item {
            background: rgba(255, 255, 255, 0.7);
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
            font-weight: bold;
        }
        .error {
            color: #e74c3c;
            padding: 15px;
            background: #ffeaa7;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        .error.active {
            display: block;
        }
        .loading {
            display: none;
            margin-top: 20px;
            color: #667eea;
            font-size: 1.2em;
        }
        .loading.active {
            display: block;
        }
        .api-note {
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 10px;
            font-size: 0.85em;
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
        <div class="loading" id="loading">⏳ Fetching weather data...</div>
        <div class="error" id="error"></div>
        <div class="api-note" id="apiNote"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="weather-icon" id="weatherIcon"></div>
            <div class="city-name" id="cityName"></div>
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

                document.getElementById('cityName').textContent = data.city + ', ' + data.country;
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';
                document.getElementById('tempMin').textContent = data.temp_min + '°C';
                document.getElementById('tempMax').textContent = data.temp_max + '°C';

                const mainWeather = data.main_weather || 'Clear';
                document.getElementById('weatherIcon').textContent = weatherEmojis[mainWeather] || '🌡️';

                weatherInfo.classList.add('active');
            } catch (err) {
                loading.classList.remove('active');
                showError('Failed to connect to server. Please try again.');
            }
        }

        function showError(message) {
            const error = document.getElementById('error');
            error.textContent = message;
            error.classList.add('active');
        }

        document.getElementById('cityInput').focus();
    </script>
</body>
</html>"""

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

DEMO_DATA = {
    "london": {"city": "London", "country": "GB", "temperature": 15, "feels_like": 13, "temp_min": 12, "temp_max": 18, "humidity": 72, "pressure": 1013, "wind_speed": 5.1, "description": "overcast clouds", "main_weather": "Clouds"},
    "new york": {"city": "New York", "country": "US", "temperature": 22, "feels_like": 21, "temp_min": 19, "temp_max": 25, "humidity": 55, "pressure": 1015, "wind_speed": 3.6, "description": "partly cloudy", "main_weather": "Clouds"},
    "tokyo": {"city": "Tokyo", "country": "JP", "temperature": 28, "feels_like": 31, "temp_min": 26, "temp_max": 30, "humidity": 80, "pressure": 1008, "wind_speed": 2.1, "description": "light rain", "main_weather": "Rain"},
    "paris": {"city": "Paris", "country": "FR", "temperature": 18, "feels_like": 17, "temp_min": 15, "temp_max": 21, "humidity": 65, "pressure": 1018, "wind_speed": 4.2, "description": "clear sky", "main_weather": "Clear"},
    "sydney": {"city": "Sydney", "country": "AU", "temperature": 20, "feels_like": 19, "temp_min": 17, "temp_max": 23, "humidity": 60, "pressure": 1020, "wind_speed": 6.7, "description": "sunny", "main_weather": "Clear"},
    "mumbai": {"city": "Mumbai", "country": "IN", "temperature": 32, "feels_like": 38, "temp_min": 30, "temp_max": 34, "humidity": 85, "pressure": 1006, "wind_speed": 4.5, "description": "haze", "main_weather": "Haze"},
    "berlin": {"city": "Berlin", "country": "DE", "temperature": 16, "feels_like": 14, "temp_min": 13, "temp_max": 19, "humidity": 68, "pressure": 1012, "wind_speed": 3.8, "description": "scattered clouds", "main_weather": "Clouds"},
    "moscow": {"city": "Moscow", "country": "RU", "temperature": 8, "feels_like": 5, "temp_min": 5, "temp_max": 11, "humidity": 75, "pressure": 1010, "wind_speed": 5.5, "description": "overcast clouds", "main_weather": "Clouds"},
    "dubai": {"city": "Dubai", "country": "AE", "temperature": 38, "feels_like": 42, "temp_min": 35, "temp_max": 41, "humidity": 45, "pressure": 1005, "wind_speed": 3.2, "description": "clear sky", "main_weather": "Clear"},
    "toronto": {"city": "Toronto", "country": "CA", "temperature": 19, "feels_like": 18, "temp_min": 16, "temp_max": 22, "humidity": 58, "pressure": 1016, "wind_speed": 4.1, "description": "few clouds", "main_weather": "Clouds"},
    "beijing": {"city": "Beijing", "country": "CN", "temperature": 25, "feels_like": 26, "temp_min": 22, "temp_max": 28, "humidity": 50, "pressure": 1011, "wind_speed": 2.8, "description": "haze", "main_weather": "Haze"},
    "cairo": {"city": "Cairo", "country": "EG", "temperature": 35, "feels_like": 33, "temp_min": 32, "temp_max": 38, "humidity": 25, "pressure": 1009, "wind_speed": 4.0, "description": "clear sky", "main_weather": "Clear"},
    "rome": {"city": "Rome", "country": "IT", "temperature": 24, "feels_like": 23, "temp_min": 21, "temp_max": 27, "humidity": 55, "pressure": 1017, "wind_speed": 3.5, "description": "sunny", "main_weather": "Clear"},
    "los angeles": {"city": "Los Angeles", "country": "US", "temperature": 26, "feels_like": 25, "temp_min": 23, "temp_max": 29, "humidity": 40, "pressure": 1014, "wind_speed": 2.5, "description": "clear sky", "main_weather": "Clear"},
    "chicago": {"city": "Chicago", "country": "US", "temperature": 17, "feels_like": 15, "temp_min": 14, "temp_max": 20, "humidity": 62, "pressure": 1013, "wind_speed": 6.2, "description": "partly cloudy", "main_weather": "Clouds"},
    "san francisco": {"city": "San Francisco", "country": "US", "temperature": 16, "feels_like": 15, "temp_min": 13, "temp_max": 19, "humidity": 78
import http.server
import json
import urllib.request
import urllib.parse
import urllib.error
import socketserver
import webbrowser
import threading

PORT = 8080

API_KEY = "demo"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

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
            animation: fadeIn 0.5s ease;
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
            font-size: 4em;
            margin: 10px 0;
        }
        .temperature {
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
        }
        .description {
            font-size: 1.2em;
            color: #666;
            text-transform: capitalize;
            margin: 10px 0;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .detail-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
        }
        .detail-label {
            font-size: 0.85em;
            color: #999;
            margin-bottom: 5px;
        }
        .detail-value {
            font-size: 1.2em;
            color: #333;
            font-weight: 600;
        }
        .error {
            display: none;
            color: #e74c3c;
            text-align: center;
            padding: 15px;
            background: #ffeaea;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            font-size: 1.2em;
            padding: 20px;
        }
        .api-note {
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 10px;
            font-size: 0.85em;
            color: #856404;
        }
        .api-note a {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ Weather App</h1>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." autofocus>
            <button onclick="getWeather()">Search</button>
        </div>
        <div class="error" id="error"></div>
        <div class="loading" id="loading">⏳ Fetching weather data...</div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <div class="weather-icon" id="weatherIcon"></div>
            <div class="temperature" id="temperature"></div>
            <div class="description" id="description"></div>
            <div class="details">
                <div class="detail-box">
                    <div class="detail-label">Feels Like</div>
                    <div class="detail-value" id="feelsLike"></div>
                </div>
                <div class="detail-box">
                    <div class="detail-label">Humidity</div>
                    <div class="detail-value" id="humidity"></div>
                </div>
                <div class="detail-box">
                    <div class="detail-label">Wind Speed</div>
                    <div class="detail-value" id="wind"></div>
                </div>
                <div class="detail-box">
                    <div class="detail-label">Pressure</div>
                    <div class="detail-value" id="pressure"></div>
                </div>
            </div>
        </div>
        <div class="api-note" id="apiNote">
            <strong>Note:</strong> To use this app, get a free API key from 
            <a href="https://openweathermap.org/api" target="_blank">OpenWeatherMap</a> 
            and set it in the app. Without a valid key, the app uses sample data for demonstration.
        </div>
    </div>
    <script>
        const cityInput = document.getElementById('cityInput');
        cityInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') getWeather();
        });

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

        async function getWeather() {
            const city = cityInput.value.trim();
            if (!city) {
                showError('Please enter a city name.');
                return;
            }

            const weatherInfo = document.getElementById('weatherInfo');
            const error = document.getElementById('error');
            const loading = document.getElementById('loading');

            error.style.display = 'none';
            weatherInfo.style.display = 'none';
            loading.style.display = 'block';

            try {
                const response = await fetch('/api/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.style.display = 'none';

                if (data.error) {
                    showError(data.error);
                    return;
                }

                document.getElementById('cityName').textContent = data.city + ', ' + data.country;
                document.getElementById('weatherIcon').textContent = weatherEmojis[data.main] || '🌡️';
                document.getElementById('temperature').textContent = data.temp + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';

                weatherInfo.style.display = 'block';
            } catch (err) {
                loading.style.display = 'none';
                showError('Failed to connect to weather service. Check your connection.');
            }
        }

        function showError(msg) {
            const error = document.getElementById('error');
            error.textContent = msg;
            error.style.display = 'block';
        }
    </script>
</body>
</html>"""

SAMPLE_WEATHER_DATA = {
    "london": {
        "city": "London", "country": "GB", "temp": 15, "feels_like": 13,
        "humidity": 72, "pressure": 1013, "wind_speed": 5.1,
        "main": "Clouds", "description": "overcast clouds"
    },
    "new york": {
        "city": "New York", "country": "US", "temp": 22, "feels_like": 21,
        "humidity": 60, "pressure": 1015, "wind_speed": 3.6,
        "main": "Clear", "description": "clear sky"
    },
    "tokyo": {
        "city": "Tokyo", "country": "JP", "temp": 28, "feels_like": 31,
        "humidity": 80, "pressure": 1008, "wind_speed": 2.1,
        "main": "Rain", "description": "light rain"
    },
    "paris": {
        "city": "Paris", "country": "FR", "temp": 18, "feels_like": 17,
        "humidity": 65, "pressure": 1020, "wind_speed": 4.5,
        "main": "Clouds", "description": "scattered clouds"
    },
    "sydney": {
        "city": "Sydney", "country": "AU", "temp": 20, "feels_like": 19,
        "humidity": 55, "pressure": 1022, "wind_speed": 6.2,
        "main": "Clear", "description": "clear sky"
    },
    "mumbai": {
        "city": "Mumbai", "country": "IN", "temp": 32, "feels_like": 38,
        "humidity": 85, "pressure": 1005, "wind_speed": 3.1,
        "main": "Rain", "description": "heavy intensity rain"
    },
    "berlin": {
        "city": "Berlin", "country": "DE", "temp": 14, "feels_like": 12,
        "humidity": 68, "pressure": 1018, "wind_speed": 4.8,
        "main": "Clouds", "description": "broken clouds"
    },
    "moscow": {
        "city": "Moscow", "country": "RU", "temp": 5, "feels_like": 1,
        "humidity": 75, "pressure": 1010, "wind_speed": 7.0,
        "main": "Snow", "description": "light snow"
    },
    "dubai": {
        "city": "Dubai", "country": "AE", "temp": 40, "feels_like": 45,
        "humidity": 30, "pressure": 1000, "wind_speed": 5.5,
        "main": "Clear", "description": "clear sky"
    },
    "toronto": {
        "city": "Toronto", "country": "CA", "temp": 10, "feels_like": 7,
        "humidity": 58, "pressure": 1016, "wind_speed": 6.0,
        "main": "Clouds", "description": "few clouds"
    },
    "los angeles": {
        "city": "Los Angeles", "country": "US", "temp": 25, "feels_like": 24,
        "humidity": 45, "pressure": 1017, "wind_speed": 3.0,
        "main": "Clear", "description": "clear sky"
    },
    "chicago": {
        "city": "Chicago", "country": "US", "temp": 12, "feels_like": 9,
        "humidity": 62, "pressure": 1014, "wind_speed": 8.2,
        "main": "Clouds", "description": "overcast clouds"
    },
    "beijing": {
        "city": "Beijing", "country": "CN", "temp": 26, "feels_like": 27,
        "humidity": 50, "pressure": 1011, "wind_speed": 2.5,
        "main": "Haze", "description": "haze"
    },
    "cairo": {
        "city": "Cairo", "country": "EG", "temp": 35, "feels_like": 33,
        "humidity": 20, "pressure": 1012, "wind_speed": 4.0,
        "main": "Clear", "description": "clear sky"
    },
    "rome": {
        "city": "Rome", "country": "IT", "temp": 24, "feels_like": 23,
        "humidity": 50, "pressure": 1019, "wind_speed": 3.2,
        "main": "Clear", "description": "clear sky"
    },
    "san francisco": {
        "city": "San Francisco", "country": "US", "temp": 16, "feels_like": 15,
        "humidity": 78, "pressure": 1018, "wind_speed": 5.8,
        "main": "Mist", "description": "mist"
    },
    "singapore": {
        "city": "Singapore", "country": "SG", "temp": 30, "feels_like": 35,
        "humidity": 88, "pressure": 1009, "wind_speed": 2.0,
        "main": "Thunderstorm", "description": "thunderstorm with rain"
    },
    "seoul": {
        "city": "Seoul", "country": "KR", "temp": 19, "feels_like": 18,
        "humidity": 55, "pressure": 1015, "wind_speed": 3.5,
        "main": "Clouds", "description": "few clouds"
    },
    "amsterdam": {
        "city": "Amsterdam", "country": "NL", "temp": 13, "feels_like": 11,
        "humidity": 80, "pressure": 1014, "wind_speed": 6.5,
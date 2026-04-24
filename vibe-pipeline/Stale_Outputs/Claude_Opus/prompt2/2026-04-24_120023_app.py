import http.server
import json
import urllib.request
import urllib.parse
import urllib.error
import os

HOST = "0.0.0.0"
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        .weather-icon {
            font-size: 4em;
            margin: 10px 0;
        }
        .temperature {
            font-size: 3em;
            color: #667eea;
            font-weight: bold;
        }
        .description {
            font-size: 1.2em;
            color: #666;
            text-transform: capitalize;
            margin: 10px 0;
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
        .error-message {
            display: none;
            color: #e74c3c;
            margin-top: 15px;
            font-size: 1.1em;
        }
        .error-message.active {
            display: block;
        }
        .loading {
            display: none;
            margin-top: 20px;
            color: #667eea;
            font-size: 1.1em;
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
        <div class="loading" id="loading">Loading...</div>
        <div class="error-message" id="error"></div>
        <div class="api-note" id="apiNote"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <div class="weather-icon" id="weatherIcon"></div>
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
            'Tornado': '🌪️',
        };

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) {
                showError('Please enter a city name.');
                return;
            }

            const weatherInfo = document.getElementById('weatherInfo');
            const errorEl = document.getElementById('error');
            const loading = document.getElementById('loading');
            const apiNote = document.getElementById('apiNote');

            weatherInfo.classList.remove('active');
            errorEl.classList.remove('active');
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
                document.getElementById('weatherIcon').textContent = weatherEmojis[data.main] || '🌡️';
                document.getElementById('temperature').textContent = data.temp + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind + ' m/s';
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';

                weatherInfo.classList.add('active');

            } catch (err) {
                loading.classList.remove('active');
                showError('Failed to fetch weather data. Please try again.');
            }
        }

        function showError(msg) {
            const errorEl = document.getElementById('error');
            errorEl.textContent = msg;
            errorEl.classList.add('active');
        }
    </script>
</body>
</html>"""


DEMO_WEATHER_DATA = {
    "london": {"city": "London", "country": "GB", "temp": 15.2, "feels_like": 14.1, "humidity": 72, "wind": 5.1, "main": "Clouds", "description": "overcast clouds"},
    "new york": {"city": "New York", "country": "US", "temp": 22.5, "feels_like": 21.8, "humidity": 55, "wind": 3.6, "main": "Clear", "description": "clear sky"},
    "tokyo": {"city": "Tokyo", "country": "JP", "temp": 28.3, "feels_like": 30.1, "humidity": 78, "wind": 2.1, "main": "Rain", "description": "light rain"},
    "paris": {"city": "Paris", "country": "FR", "temp": 18.7, "feels_like": 17.9, "humidity": 65, "wind": 4.2, "main": "Clouds", "description": "scattered clouds"},
    "sydney": {"city": "Sydney", "country": "AU", "temp": 20.1, "feels_like": 19.5, "humidity": 60, "wind": 6.7, "main": "Clear", "description": "clear sky"},
    "berlin": {"city": "Berlin", "country": "DE", "temp": 14.8, "feels_like": 13.5, "humidity": 70, "wind": 3.9, "main": "Clouds", "description": "broken clouds"},
    "moscow": {"city": "Moscow", "country": "RU", "temp": 5.2, "feels_like": 2.1, "humidity": 80, "wind": 4.5, "main": "Snow", "description": "light snow"},
    "dubai": {"city": "Dubai", "country": "AE", "temp": 38.5, "feels_like": 42.0, "humidity": 45, "wind": 3.2, "main": "Clear", "description": "clear sky"},
    "mumbai": {"city": "Mumbai", "country": "IN", "temp": 32.1, "feels_like": 36.5, "humidity": 85, "wind": 4.8, "main": "Rain", "description": "moderate rain"},
    "beijing": {"city": "Beijing", "country": "CN", "temp": 25.6, "feels_like": 24.8, "humidity": 50, "wind": 2.5, "main": "Haze", "description": "haze"},
    "cairo": {"city": "Cairo", "country": "EG", "temp": 35.0, "feels_like": 34.2, "humidity": 30, "wind": 3.0, "main": "Clear", "description": "clear sky"},
    "rome": {"city": "Rome", "country": "IT", "temp": 24.3, "feels_like": 23.8, "humidity": 55, "wind": 2.8, "main": "Clear", "description": "clear sky"},
    "toronto": {"city": "Toronto", "country": "CA", "temp": 12.5, "feels_like": 10.9, "humidity": 68, "wind": 5.5, "main": "Clouds", "description": "few clouds"},
    "san francisco": {"city": "San Francisco", "country": "US", "temp": 16.8, "feels_like": 15.9, "humidity": 75, "wind": 6.2, "main": "Mist", "description": "mist"},
    "los angeles": {"city": "Los Angeles", "country": "US", "temp": 26.4, "feels_like": 25.8, "humidity": 40, "wind": 2.3, "main": "Clear", "description": "clear sky"},
    "chicago": {"city": "Chicago", "country": "US", "temp": 10.5, "feels_like": 8.2, "humidity": 62, "wind": 7.1, "main": "Clouds", "description": "overcast clouds"},
    "singapore": {"city": "Singapore", "country": "SG", "temp": 30.5, "feels_like": 34.2, "humidity": 88, "wind": 1.5, "main": "Thunderstorm", "description": "thunderstorm with rain"},
    "seoul": {"city": "Seoul", "country": "KR", "temp": 19.8, "feels_like": 18.5, "humidity": 58, "wind": 3.1, "main": "Clear", "description": "clear sky"},
    "bangkok": {"city": "Bangkok", "country": "TH", "temp": 33.2, "feels_like": 38.0, "humidity": 82, "wind": 1.8, "main": "Clouds", "description": "scattered clouds"},
    "amsterdam": {"city": "Amsterdam", "country": "NL", "temp": 13.5, "feels_like": 11.8, "humidity": 78, "wind": 5.8, "main": "Drizzle", "description": "light drizzle"},
}


def fetch_weather_from_api(city):
    if not API_KEY:
        return None

    try:
        encoded_city = urllib.parse.quote(city)
        url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_city}&appid={API_KEY}&units=metric"
        req = urllib.request.Request(url, headers={"User-Agent": "WeatherApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        result = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temp": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "humidity": data["main"]["humidity"],
            "wind": round(data["wind"]["speed"], 1),
            "main": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
        }
        return result
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": "City not found. Please check the spelling and try again."}
        elif e.code == 401:
            return None
        else:
            return {"error": f"API error: {e.code}"}
    except Exception:
        return None


def get_demo_weather(city):
    city_lower = city.lower().strip()
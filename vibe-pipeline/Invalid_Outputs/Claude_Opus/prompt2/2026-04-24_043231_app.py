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
            border-radius: 25px;
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
            border-radius: 25px;
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
            width: 100px;
            height: 100px;
        }
        .temperature {
            font-size: 3em;
            color: #667eea;
            font-weight: bold;
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
            color: #999;
            font-size: 0.9em;
        }
        .detail-value {
            color: #333;
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 5px;
        }
        .error {
            color: #e74c3c;
            margin-top: 20px;
            font-size: 1.1em;
            display: none;
        }
        .error.active {
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
        .api-notice {
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 10px;
            color: #856404;
            font-size: 0.85em;
            display: none;
        }
        .api-notice.active {
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

            const weatherInfo = document.getElementById('weatherInfo');
            const error = document.getElementById('error');
            const loading = document.getElementById('loading');
            const apiNotice = document.getElementById('apiNotice');

            weatherInfo.classList.remove('active');
            error.classList.remove('active');
            apiNotice.classList.remove('active');
            loading.classList.add('active');

            try {
                const response = await fetch('/api/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.classList.remove('active');

                if (data.error) {
                    error.textContent = data.error;
                    error.classList.add('active');
                    if (data.notice) {
                        apiNotice.textContent = data.notice;
                        apiNotice.classList.add('active');
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


def fetch_weather(city):
    if not API_KEY:
        return generate_demo_weather(city)

    base_url = "https://api.openweathermap.org/data/2.5/weather"
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

        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "description": data["weather"][0]["description"],
            "icon": f"https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png",
            "humidity": data["main"]["humidity"],
            "wind_speed": round(data["wind"]["speed"], 1)
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"City '{city}' not found. Please check the spelling and try again."}
        elif e.code == 401:
            return {"error": "Invalid API key.", "notice": "Please set a valid OPENWEATHERMAP_API_KEY environment variable."}
        else:
            return {"error": f"API error: {e.code}"}
    except urllib.error.URLError:
        return {"error": "Could not connect to weather service. Check your internet connection."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def generate_demo_weather(city):
    import hashlib
    import math

    city_lower = city.lower().strip()

    known_cities = {
        "london": {"country": "GB", "base_temp": 12, "humidity": 75},
        "new york": {"country": "US", "base_temp": 15, "humidity": 60},
        "tokyo": {"country": "JP", "base_temp": 18, "humidity": 65},
        "paris": {"country": "FR", "base_temp": 14, "humidity": 70},
        "sydney": {"country": "AU", "base_temp": 22, "humidity": 55},
        "moscow": {"country": "RU", "base_temp": 5, "humidity": 72},
        "dubai": {"country": "AE", "base_temp": 35, "humidity": 40},
        "mumbai": {"country": "IN", "base_temp": 30, "humidity": 80},
        "berlin": {"country": "DE", "base_temp": 11, "humidity": 68},
        "rome": {"country": "IT", "base_temp": 18, "humidity": 62},
        "beijing": {"country": "CN", "base_temp": 14, "humidity": 50},
        "cairo": {"country": "EG", "base_temp": 28, "humidity": 35},
        "toronto": {"country": "CA", "base_temp": 10, "humidity": 65},
        "singapore": {"country": "SG", "base_temp": 31, "humidity": 85},
        "los angeles": {"country": "US", "base_temp": 22, "humidity": 45},
        "chicago": {"country": "US", "base_temp": 12, "humidity": 63},
        "san francisco": {"country": "US", "base_temp": 16, "humidity": 72},
        "seattle": {"country": "US", "base_temp": 13, "humidity": 78},
        "amsterdam": {"country": "NL", "base_temp": 11, "humidity": 80},
        "bangkok": {"country": "TH", "base_temp": 32, "humidity": 75},
        "seoul": {"country": "KR", "base_temp": 13, "humidity": 60},
        "mexico city": {"country": "MX", "base_temp": 20, "humidity": 50},
        "istanbul": {"country": "TR", "base_temp": 16, "humidity": 65},
        "buenos aires": {"country": "AR", "base_temp": 19, "humidity": 68},
        "lagos": {"country": "NG", "base_temp": 28, "humidity": 78},
        "nairobi": {"country": "KE", "base_temp": 20, "humidity": 60},
        "lima": {"country": "PE", "base_temp": 19, "humidity": 82},
        "rio de janeiro": {"country": "BR", "base_temp": 26, "humidity": 77},
        "cape town": {"country": "ZA", "base_temp": 18, "humidity": 65},
        "delhi": {"country": "IN", "base_temp": 28, "humidity": 55},
        "hong kong": {"country": "HK", "base_temp": 25, "humidity": 78},
        "madrid": {"country": "ES", "base_temp": 17, "humidity": 45},
        "lisbon": {"country": "PT", "base_temp": 18, "humidity": 70},
        "vienna": {"country": "AT", "base_temp": 12, "humidity": 65},
        "zurich": {"country": "CH", "base_temp": 10, "humidity": 72},
        "stockholm": {"country": "SE", "base_temp": 8, "humidity": 74},
        "oslo": {"country": "NO", "base_temp": 6, "humidity": 72},
        "helsinki": {"country": "FI", "base_temp": 5, "humidity": 76},
        "warsaw": {"country": "PL", "base_temp": 10, "humidity": 70},
        "prague": {"country": "CZ", "base_temp": 11, "humidity": 68},
        "athens": {"country": "GR", "base_temp": 20, "humidity": 55},
    }

    city_info = known_cities.get(city_lower)

    if city_info is None:
        hash_val = int(hashlib.md5(city_lower.encode()).hexdigest(), 16)
        base_temp = (hash_val % 35) + 2
        humidity = (hash_val % 50) + 30
        country_codes = ["US", "GB", "DE", "FR", "ES", "IT", "BR", "IN", "CN", "AU", "JP", "CA", "MX", "RU", "KR"]
        country = country_codes[hash_val % len(country_codes)]
    else:
        base_temp = city_info["base_temp"]
        humidity = city_info["humidity"]
        country = city_info["country"]

    import time
    time_seed = int(time.time()) // 3600
    variation = ((time_seed * 7 + len(city)) % 11) - 5
    temp = base_temp + variation
    feels_like = temp - 2

    weather_conditions = [
        {"desc": "clear sky", "icon": "01d"},
        {"desc": "few clouds", "icon": "02
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
            color: #999;
            font-size: 0.9em;
        }
        .detail-value {
            color: #333;
            font-size: 1.2em;
            font-weight: bold;
        }
        .error {
            color: #e74c3c;
            font-size: 1.1em;
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
        .weather-emoji {
            font-size: 5em;
            margin: 10px 0;
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
        <div class="loading" id="loading">Loading...</div>
        <div class="error" id="error"></div>
        <div id="apiNotice" class="api-notice"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <div class="weather-emoji" id="weatherEmoji"></div>
            <div class="temperature" id="temperature"></div>
            <div class="description" id="description"></div>
            <div class="details">
                <div class="detail-item">
                    <div class="detail-label">Humidity</div>
                    <div class="detail-value" id="humidity"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Wind</div>
                    <div class="detail-value" id="wind"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Feels Like</div>
                    <div class="detail-value" id="feelsLike"></div>
                </div>
            </div>
            <div class="details">
                <div class="detail-item">
                    <div class="detail-label">Pressure</div>
                    <div class="detail-value" id="pressure"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Visibility</div>
                    <div class="detail-value" id="visibility"></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Clouds</div>
                    <div class="detail-value" id="clouds"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('cityInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                getWeather();
            }
        });

        function getWeatherEmoji(code) {
            if (code >= 200 && code < 300) return '⛈️';
            if (code >= 300 && code < 400) return '🌦️';
            if (code >= 500 && code < 600) return '🌧️';
            if (code >= 600 && code < 700) return '❄️';
            if (code >= 700 && code < 800) return '🌫️';
            if (code === 800) return '☀️';
            if (code === 801) return '🌤️';
            if (code === 802) return '⛅';
            if (code >= 803) return '☁️';
            return '🌡️';
        }

        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) {
                showError('Please enter a city name.');
                return;
            }

            const weatherInfo = document.getElementById('weatherInfo');
            const errorDiv = document.getElementById('error');
            const loading = document.getElementById('loading');
            const apiNotice = document.getElementById('apiNotice');

            weatherInfo.classList.remove('active');
            errorDiv.classList.remove('active');
            apiNotice.classList.remove('active');
            loading.classList.add('active');

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.classList.remove('active');

                if (data.error) {
                    showError(data.error);
                    if (data.notice) {
                        apiNotice.textContent = data.notice;
                        apiNotice.classList.add('active');
                    }
                    return;
                }

                document.getElementById('cityName').textContent = data.city + ', ' + data.country;
                document.getElementById('weatherEmoji').textContent = getWeatherEmoji(data.weather_code);
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';
                document.getElementById('visibility').textContent = data.visibility + ' km';
                document.getElementById('clouds').textContent = data.clouds + '%';

                weatherInfo.classList.add('active');
            } catch (err) {
                loading.classList.remove('active');
                showError('Failed to fetch weather data. Please try again.');
            }
        }

        function showError(msg) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = msg;
            errorDiv.classList.add('active');
        }
    </script>
</body>
</html>"""


def fetch_weather_from_api(city):
    if not API_KEY:
        return None, "No API key configured. Set the OPENWEATHERMAP_API_KEY environment variable."

    encoded_city = urllib.parse.quote(city)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_city}&appid={API_KEY}&units=metric"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WeatherApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data, None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, f"City '{city}' not found. Please check the spelling and try again."
        elif e.code == 401:
            return None, "Invalid API key. Please check your OPENWEATHERMAP_API_KEY."
        else:
            return None, f"API error (HTTP {e.code}). Please try again later."
    except urllib.error.URLError:
        return None, "Could not connect to weather service. Check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def fetch_weather_fallback(city):
    encoded_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WeatherApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data, None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, f"City '{city}' not found."
        return None, f"Weather service error (HTTP {e.code})."
    except urllib.error.URLError:
        return None, "Could not connect to weather service. Check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def parse_openweathermap(data):
    visibility_km = round(data.get("visibility", 0) / 1000, 1)
    return {
        "city": data.get("name", "Unknown"),
        "country": data.get("sys", {}).get("country", ""),
        "temperature": round(data["main"]["temp"], 1),
        "feels_like": round(data["main"]["feels_like"], 1),
        "description": data["weather"][0]["description"] if data.get("weather") else "N/A",
        "weather_code": data["weather"][0]["id"] if data.get("weather") else 800,
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": round(data.get("wind", {}).get("speed", 0), 1),
        "visibility": visibility_km,
        "clouds": data.get("clouds", {}).get("all", 0),
    }


def parse_wttr(data, city):
    current = data.get("current_condition", [{}])[0]
    nearest = data.get("nearest_area", [{}])[0]

    area_name = city
    country = ""
    if nearest:
        area_names = nearest.get("areaName", [{}])
        if area_names:
            area_name = area_names[0].get("value", city)
        countries = nearest.get("country", [{}])
        if countries:
            country = countries[0].get("value", "")

    temp_c = current.get("temp_C", "0")
    feels_like = current.get("FeelsLikeC", "0")
    humidity = current.get("humidity", "0")
    pressure = current.get("pressure", "0")
    wind_speed_kmph = current.get("windspeedKmph", "0")
    visibility = current.get("visibility", "0")
    cloud_cover = current.get("cloudcover", "0")

    weather_desc_list = current.get("weatherDesc", [{}])
    description = weather_desc_list[0].get("value", "N/A") if weather_desc_list else "N/A"

    weather_code_str = current.get("weatherCode", "113")
    try:
        wttr_code = int(weather_code_str)
    except ValueError:
        wttr_code = 113

    owm_code = 800
    if wttr_code in (386, 389, 392, 395, 200):
        owm_code = 200
    elif wttr_code in (263, 266, 281, 284, 293, 296):
        owm_code = 300
    elif wttr_code in (176, 299, 302, 305, 308, 311, 314, 317, 320, 353, 356, 359, 362, 365):
        owm_code = 500
    elif wttr_code in (179, 182, 185, 227, 230, 323, 326, 329, 332, 335, 338, 368, 371, 374, 377):
        owm_code = 600
    elif wttr_code in (143, 248, 260):
        owm_code =
import http.server
import json
import urllib.request
import urllib.parse
import urllib.error
import socketserver
import webbrowser
import threading
import os

PORT = 8080

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
            align-items: center;
            justify-content: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
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
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .city-name {
            font-size: 24px;
            color: #333;
            margin-bottom: 5px;
        }
        .country {
            font-size: 14px;
            color: #888;
            margin-bottom: 15px;
        }
        .weather-icon {
            font-size: 64px;
            margin: 10px 0;
        }
        .temperature {
            font-size: 52px;
            font-weight: bold;
            color: #333;
        }
        .description {
            font-size: 18px;
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
        .detail-card {
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
        .error-message {
            display: none;
            color: #e74c3c;
            background: #ffeaea;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .loading {
            display: none;
            color: #667eea;
            font-size: 18px;
            margin: 20px 0;
        }
        .api-notice {
            display: none;
            background: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-size: 13px;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ Weather App</h1>
        <div class="api-notice" id="apiNotice">
            <strong>Note:</strong> No API key detected. Set the environment variable
            <code>OPENWEATHERMAP_API_KEY</code> with your free key from
            <a href="https://openweathermap.org/api" target="_blank">openweathermap.org</a>
            and restart the app. The app will use demo data until then.
        </div>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." autofocus>
            <button onclick="getWeather()">Search</button>
        </div>
        <div class="error-message" id="errorMsg"></div>
        <div class="loading" id="loading">⏳ Fetching weather data...</div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <div class="country" id="country"></div>
            <div class="weather-icon" id="weatherIcon"></div>
            <div class="temperature" id="temperature"></div>
            <div class="description" id="description"></div>
            <div class="details">
                <div class="detail-card">
                    <div class="detail-label">Feels Like</div>
                    <div class="detail-value" id="feelsLike"></div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Humidity</div>
                    <div class="detail-value" id="humidity"></div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Wind Speed</div>
                    <div class="detail-value" id="wind"></div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Pressure</div>
                    <div class="detail-value" id="pressure"></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const HAS_API_KEY = APIKEY_PLACEHOLDER;

        if (!HAS_API_KEY) {
            document.getElementById('apiNotice').style.display = 'block';
        }

        document.getElementById('cityInput').addEventListener('keypress', function(e) {
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
            const city = document.getElementById('cityInput').value.trim();
            if (!city) {
                showError('Please enter a city name.');
                return;
            }

            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('weatherInfo').style.display = 'none';
            document.getElementById('loading').style.display = 'block';

            try {
                const response = await fetch('/api/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';

                if (data.error) {
                    showError(data.error);
                    return;
                }

                document.getElementById('cityName').textContent = data.city;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').textContent = weatherEmojis[data.main] || '🌡️';
                document.getElementById('temperature').textContent = Math.round(data.temp) + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = Math.round(data.feels_like) + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';

                document.getElementById('weatherInfo').style.display = 'block';
            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                showError('Failed to fetch weather data. Please try again.');
            }
        }

        function showError(msg) {
            const el = document.getElementById('errorMsg');
            el.textContent = msg;
            el.style.display = 'block';
        }
    </script>
</body>
</html>"""

DEMO_DATA = {
    "london": {"city": "London", "country": "United Kingdom", "temp": 15.2, "feels_like": 13.8, "humidity": 72, "pressure": 1013, "wind_speed": 4.1, "description": "overcast clouds", "main": "Clouds"},
    "new york": {"city": "New York", "country": "United States", "temp": 22.5, "feels_like": 21.0, "humidity": 55, "pressure": 1018, "wind_speed": 3.6, "description": "clear sky", "main": "Clear"},
    "tokyo": {"city": "Tokyo", "country": "Japan", "temp": 28.3, "feels_like": 30.1, "humidity": 78, "pressure": 1008, "wind_speed": 2.5, "description": "light rain", "main": "Rain"},
    "paris": {"city": "Paris", "country": "France", "temp": 18.7, "feels_like": 17.2, "humidity": 65, "pressure": 1015, "wind_speed": 3.2, "description": "scattered clouds", "main": "Clouds"},
    "sydney": {"city": "Sydney", "country": "Australia", "temp": 20.1, "feels_like": 19.5, "humidity": 60, "pressure": 1020, "wind_speed": 5.0, "description": "few clouds", "main": "Clouds"},
    "berlin": {"city": "Berlin", "country": "Germany", "temp": 14.3, "feels_like": 12.8, "humidity": 70, "pressure": 1012, "wind_speed": 4.5, "description": "broken clouds", "main": "Clouds"},
    "moscow": {"city": "Moscow", "country": "Russia", "temp": 5.2, "feels_like": 2.1, "humidity": 80, "pressure": 1005, "wind_speed": 6.0, "description": "snow", "main": "Snow"},
    "mumbai": {"city": "Mumbai", "country": "India", "temp": 32.4, "feels_like": 36.0, "humidity": 85, "pressure": 1006, "wind_speed": 3.8, "description": "haze", "main": "Haze"},
    "cairo": {"city": "Cairo", "country": "Egypt", "temp": 35.6, "feels_like": 34.2, "humidity": 25, "pressure": 1010, "wind_speed": 4.2, "description": "clear sky", "main": "Clear"},
    "beijing": {"city": "Beijing", "country": "China", "temp": 26.8, "feels_like": 27.5, "humidity": 50, "pressure": 1016, "wind_speed": 2.8, "description": "mist", "main": "Mist"},
    "los angeles": {"city": "Los Angeles", "country": "United States", "temp": 25.3, "feels_like": 24.8, "humidity": 40, "pressure": 1017, "wind_speed": 3.0, "description": "clear sky", "main": "Clear"},
    "san francisco": {"city": "San Francisco", "country": "United States", "temp": 16.5, "feels_like": 15.0, "humidity": 75, "pressure": 1014, "wind_speed": 5.5, "description": "fog", "main": "Fog"},
    "toronto": {"city": "Toronto", "country": "Canada", "temp": 12.0, "feels_like": 10.5, "humidity": 68, "pressure": 1011, "wind_speed": 4.8, "description": "drizzle", "main": "Drizzle"},
    "dubai": {"city": "Dubai", "country": "United Arab Emirates", "temp": 40.2, "feels_like": 42.5, "humidity": 30, "pressure": 1008, "wind_speed": 3.5, "description": "clear sky", "main": "Clear"},
    "rome": {"city": "Rome", "country": "Italy", "temp": 24.1, "feels_like": 23.5, "humidity": 55, "pressure": 1016, "wind_speed": 2.9, "description": "sunny", "main": "Clear"},
}

COUNTRY_CODES = {
    "AF": "Afghanistan", "AL": "Albania", "DZ": "Algeria", "AD": "Andorra", "AO": "Angola",
    "AR": "Argentina", "AM": "Armenia", "AU": "Australia", "AT": "Austria", "AZ": "Azerbaijan",
    "BS": "Bahamas", "BH": "Bahrain", "BD": "Bangladesh", "BB": "Barbados", "BY": "Belarus",
    "BE": "Belgium", "BZ": "Belize", "BJ": "Benin", "BT": "Bhutan", "BO": "Bolivia",
    "BA": "Bosnia and Herzegovina", "BW": "Botswana", "BR": "Brazil", "BN": "Brunei", "BG": "Bulgaria",
    "CA": "Canada", "CL": "Chile", "CN": "China", "CO": "Colombia", "CR": "Costa Rica",
    "HR": "Croatia", "CU": "Cuba", "CY": "Cyprus", "CZ": "Czech Republic", "DK": "Denmark",
    "DO": "Dominican Republic", "EC": "Ecuador", "EG": "Egypt", "SV": "El Salvador", "EE": "Estonia",
    "ET": "
import urllib.request
import urllib.parse
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler


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
            width: 420px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
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
            text-align: center;
        }
        .weather-info.active {
            display: block;
        }
        .city-name {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .country {
            font-size: 14px;
            color: #888;
            margin-bottom: 20px;
        }
        .weather-icon {
            width: 100px;
            height: 100px;
            margin: 0 auto;
        }
        .weather-icon img {
            width: 100%;
            height: 100%;
        }
        .temperature {
            font-size: 52px;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
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
            display: none;
            text-align: center;
            color: #e74c3c;
            padding: 20px;
            font-size: 16px;
        }
        .error.active {
            display: block;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .loading.active {
            display: block;
        }
        .api-notice {
            text-align: center;
            font-size: 12px;
            color: #999;
            margin-top: 20px;
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
        <div class="loading" id="loading">Loading weather data...</div>
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
        <div class="api-notice">Powered by OpenWeatherMap API</div>
    </div>
    <script>
        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) return;

            const weatherInfo = document.getElementById('weatherInfo');
            const error = document.getElementById('error');
            const loading = document.getElementById('loading');

            weatherInfo.classList.remove('active');
            error.classList.remove('active');
            loading.classList.add('active');

            try {
                const response = await fetch('/api/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.classList.remove('active');

                if (data.error) {
                    error.textContent = data.error;
                    error.classList.add('active');
                    return;
                }

                document.getElementById('cityName').textContent = data.name;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').innerHTML = '<img src="https://openweathermap.org/img/wn/' + data.icon + '@2x.png" alt="weather icon">';
                document.getElementById('temperature').textContent = data.temp + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';
                document.getElementById('visibility').textContent = data.visibility + ' km';
                document.getElementById('clouds').textContent = data.clouds + '%';

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
        return {"error": "API key not configured. Set OPENWEATHERMAP_API_KEY environment variable. Get a free key at https://openweathermap.org/api"}

    params = urllib.parse.urlencode({
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    })
    url = f"https://api.openweathermap.org/data/2.5/weather?{params}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        visibility_km = round(data.get("visibility", 0) / 1000, 1)

        result = {
            "name": data["name"],
            "country": data["sys"].get("country", ""),
            "temp": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": round(data["wind"]["speed"], 1),
            "description": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"],
            "visibility": visibility_km,
            "clouds": data["clouds"]["all"]
        }
        return result
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"City '{city}' not found. Please check the spelling and try again."}
        elif e.code == 401:
            return {"error": "Invalid API key. Please check your OPENWEATHERMAP_API_KEY."}
        else:
            return {"error": f"API error (HTTP {e.code}). Please try again later."}
    except urllib.error.URLError:
        return {"error": "Network error. Please check your internet connection."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


class WeatherHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path.startswith("/api/weather"):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            city = params.get("city", [""])[0]

            if not city:
                result = {"error": "Please enter a city name."}
            else:
                result = fetch_weather(city)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), WeatherHandler)

    if not API_KEY:
        print("=" * 60)
        print("WARNING: OPENWEATHERMAP_API_KEY environment variable not set!")
        print("")
        print("To use this app, you need a free API key from:")
        print("  https://openweathermap.org/api")
        print("")
        print("Then run the app like this:")
        print(f"  OPENWEATHERMAP_API_KEY=your_key_here python app.py")
        print("")
        print("Or set it in your environment:")
        print("  export OPENWEATHERMAP_API_KEY=your_key_here")
        print("=" * 60)
    else:
        print("API key configured successfully.")

    print(f"\nWeather App is running at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
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
            font-size: 4em;
            margin: 10px 0;
        }
        .temperature {
            font-size: 3em;
            color: #667eea;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .description {
            font-size: 1.2em;
            color: #666;
            text-transform: capitalize;
            margin-bottom: 20px;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .detail-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 15px;
        }
        .detail-label {
            font-size: 0.85em;
            color: #999;
            margin-bottom: 5px;
        }
        .detail-value {
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }
        .error {
            color: #e74c3c;
            font-size: 1.1em;
            margin-top: 20px;
            display: none;
        }
        .loading {
            display: none;
            margin-top: 20px;
            color: #667eea;
            font-size: 1.1em;
        }
        .api-key-notice {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 0.85em;
            color: #856404;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ Weather App</h1>
        <div class="api-key-notice" id="apiNotice">
            <strong>Note:</strong> Using wttr.in free weather service - no API key needed!
        </div>
        <div class="search-box">
            <input type="text" id="cityInput" placeholder="Enter city name..." onkeypress="if(event.key==='Enter')getWeather()">
            <button onclick="getWeather()">Search</button>
        </div>
        <div class="loading" id="loading">⏳ Fetching weather data...</div>
        <div class="error" id="error"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
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
                    <div class="detail-label">Visibility</div>
                    <div class="detail-value" id="visibility"></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        async function getWeather() {
            const city = document.getElementById('cityInput').value.trim();
            if (!city) {
                showError('Please enter a city name');
                return;
            }

            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const weatherInfo = document.getElementById('weatherInfo');

            loading.style.display = 'block';
            error.style.display = 'none';
            weatherInfo.style.display = 'none';

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.style.display = 'none';

                if (data.error) {
                    showError(data.error);
                    return;
                }

                document.getElementById('cityName').textContent = data.city + ', ' + data.country;
                document.getElementById('weatherIcon').textContent = data.icon;
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' km/h';
                document.getElementById('visibility').textContent = data.visibility + ' km';

                weatherInfo.style.display = 'block';
            } catch (err) {
                loading.style.display = 'none';
                showError('Failed to fetch weather data. Please try again.');
                console.error(err);
            }
        }

        function showError(message) {
            const error = document.getElementById('error');
            error.textContent = message;
            error.style.display = 'block';
        }
    </script>
</body>
</html>"""


def get_weather_icon(code):
    weather_icons = {
        "113": "☀️",
        "116": "⛅",
        "119": "☁️",
        "122": "☁️",
        "143": "🌫️",
        "176": "🌦️",
        "179": "🌨️",
        "182": "🌨️",
        "185": "🌨️",
        "200": "⛈️",
        "227": "🌨️",
        "230": "❄️",
        "248": "🌫️",
        "260": "🌫️",
        "263": "🌦️",
        "266": "🌦️",
        "281": "🌨️",
        "284": "🌨️",
        "293": "🌦️",
        "296": "🌧️",
        "299": "🌧️",
        "302": "🌧️",
        "305": "🌧️",
        "308": "🌧️",
        "311": "🌨️",
        "314": "🌨️",
        "317": "🌨️",
        "320": "🌨️",
        "323": "🌨️",
        "326": "🌨️",
        "329": "❄️",
        "332": "❄️",
        "335": "❄️",
        "338": "❄️",
        "350": "🌨️",
        "353": "🌦️",
        "356": "🌧️",
        "359": "🌧️",
        "362": "🌨️",
        "365": "🌨️",
        "368": "🌨️",
        "371": "❄️",
        "374": "🌨️",
        "377": "🌨️",
        "386": "⛈️",
        "389": "⛈️",
        "392": "⛈️",
        "395": "❄️",
    }
    return weather_icons.get(str(code), "🌡️")


def fetch_weather(city):
    try:
        encoded_city = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded_city}?format=j1"

        req = urllib.request.Request(url)
        req.add_header("User-Agent", "WeatherApp/1.0")

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data["current_condition"][0]
        nearest_area = data["nearest_area"][0]

        weather_code = current.get("weatherCode", "113")
        icon = get_weather_icon(weather_code)

        city_name = nearest_area["areaName"][0]["value"]
        country = nearest_area["country"][0]["value"]

        description_list = current.get("weatherDesc", [])
        description = description_list[0]["value"] if description_list else "Unknown"

        temp_c = current.get("temp_C", "N/A")
        feels_like = current.get("FeelsLikeC", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_speed = current.get("windspeedKmph", "N/A")
        visibility = current.get("visibility", "N/A")

        return {
            "city": city_name,
            "country": country,
            "icon": icon,
            "temperature": temp_c,
            "description": description,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "visibility": visibility,
        }

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"City '{city}' not found. Please check the spelling."}
        return {"error": f"HTTP error: {e.code}"}
    except urllib.error.URLError:
        return {"error": "Could not connect to weather service. Check your internet connection."}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"error": f"Error parsing weather data: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


class WeatherHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/" or parsed_path.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed_path.path == "/weather":
            query_params = urllib.parse.parse_qs(parsed_path.query)
            city = query_params.get("city", [""])[0]

            if not city:
                result = {"error": "Please provide a city name"}
            else:
                result = fetch_weather(city)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    print("=" * 50)
    print("  🌤️  Weather App")
    print("=" * 50)
    print()
    print("  This app uses wttr.in - no API key needed!")
    print()
    print(f"  Starting server on http://localhost:{PORT}")
    print(f"  Open your browser and go to http://localhost:{PORT}")
    print()
    print("  Press Ctrl+C to stop the server")
    print("=" * 50)

    with socketserver.TCPServer(("", PORT), WeatherHandler) as httpd:
        httpd.allow_reuse_address = True
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped. Goodbye! 👋")
            httpd.shutdown()


if __name__ == "__main__":
    main()
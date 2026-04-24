import http.server
import json
import urllib.request
import urllib.parse
import socketserver

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
            margin-bottom: 25px;
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
            margin-bottom: 5px;
        }
        .country {
            color: #888;
            font-size: 1em;
            margin-bottom: 15px;
        }
        .weather-icon {
            font-size: 4em;
            margin: 10px 0;
        }
        .temperature {
            font-size: 3.5em;
            color: #333;
            font-weight: bold;
        }
        .description {
            font-size: 1.2em;
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
        .detail-box {
            background: #f8f9ff;
            padding: 15px;
            border-radius: 12px;
        }
        .detail-label {
            color: #888;
            font-size: 0.85em;
            margin-bottom: 5px;
        }
        .detail-value {
            color: #333;
            font-size: 1.2em;
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
            color: #667eea;
            font-size: 1.1em;
            margin-top: 15px;
        }
        .loading::after {
            content: '';
            animation: dots 1.5s infinite;
        }
        @keyframes dots {
            0% { content: ''; }
            25% { content: '.'; }
            50% { content: '..'; }
            75% { content: '...'; }
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
        <div class="loading" id="loading">Fetching weather data</div>
        <div class="error" id="error"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <div class="country" id="country"></div>
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
    </div>
    <script>
        function getWeatherEmoji(code) {
            if (code >= 200 && code < 300) return '⛈️';
            if (code >= 300 && code < 400) return '🌧️';
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
                showError('Please enter a city name');
                return;
            }

            const weatherInfo = document.getElementById('weatherInfo');
            const error = document.getElementById('error');
            const loading = document.getElementById('loading');

            weatherInfo.style.display = 'none';
            error.style.display = 'none';
            loading.style.display = 'block';

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                loading.style.display = 'none';

                if (data.error) {
                    showError(data.error);
                    return;
                }

                document.getElementById('cityName').textContent = data.name;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').textContent = getWeatherEmoji(data.weather_id);
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('pressure').textContent = data.pressure + ' hPa';

                weatherInfo.style.display = 'block';
            } catch (err) {
                loading.style.display = 'none';
                showError('Failed to fetch weather data. Please try again.');
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


# Using wttr.in as a free weather API that requires no API key
def get_weather_data(city):
    try:
        encoded_city = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded_city}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data["current_condition"][0]
        nearest_area = data["nearest_area"][0]

        weather_code_map = {
            "113": 800, "116": 801, "119": 802, "122": 803,
            "143": 701, "176": 500, "179": 600, "182": 611,
            "185": 611, "200": 200, "227": 600, "230": 601,
            "248": 741, "260": 741, "263": 300, "266": 301,
            "281": 511, "284": 511, "293": 500, "296": 500,
            "299": 501, "302": 502, "305": 502, "308": 503,
            "311": 511, "314": 511, "317": 611, "320": 601,
            "323": 600, "326": 600, "329": 601, "332": 601,
            "335": 602, "338": 602, "350": 611, "353": 500,
            "356": 501, "359": 502, "362": 611, "365": 611,
            "368": 600, "371": 601, "374": 611, "377": 611,
            "386": 200, "389": 202, "392": 200, "395": 602,
        }

        weather_code_raw = current.get("weatherCode", "800")
        weather_id = weather_code_map.get(weather_code_raw, 800)

        result = {
            "name": nearest_area["areaName"][0]["value"],
            "country": nearest_area["country"][0]["value"],
            "temperature": current["temp_C"],
            "feels_like": current["FeelsLikeC"],
            "description": current["weatherDesc"][0]["value"],
            "humidity": current["humidity"],
            "wind_speed": round(int(current["windspeedKmph"]) / 3.6, 1),
            "pressure": current["pressure"],
            "weather_id": weather_id,
        }
        return result
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"City '{city}' not found. Please check the spelling."}
        return {"error": f"HTTP error: {e.code}"}
    except urllib.error.URLError:
        return {"error": "Could not connect to weather service. Check your internet connection."}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"error": f"City '{city}' not found or invalid response from weather service."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


class WeatherHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path.startswith("/weather"):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            city = params.get("city", [""])[0]

            if not city:
                result = {"error": "Please provide a city name"}
            else:
                result = get_weather_data(city)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), WeatherHandler) as httpd:
        print(f"🌤️  Weather App is running!")
        print(f"   Open your browser and go to: http://localhost:{PORT}")
        print(f"   Press Ctrl+C to stop the server")
        print()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
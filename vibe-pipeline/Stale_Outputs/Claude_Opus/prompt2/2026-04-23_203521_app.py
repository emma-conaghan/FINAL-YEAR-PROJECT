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
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
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
            margin-bottom: 5px;
        }
        .country {
            font-size: 1em;
            color: #888;
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
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .detail-item {
            text-align: center;
        }
        .detail-label {
            font-size: 0.85em;
            color: #888;
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
            margin-top: 20px;
            display: none;
        }
        .error.active {
            display: block;
        }
        .loading {
            display: none;
            margin-top: 20px;
            color: #666;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
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
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Fetching weather data...</p>
        </div>
        <div class="error" id="error"></div>
        <div class="api-note" id="apiNote"></div>
        <div class="weather-info" id="weatherInfo">
            <div class="city-name" id="cityName"></div>
            <div class="country" id="country"></div>
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
            'Snow': '🌨️',
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

            document.getElementById('loading').classList.add('active');
            document.getElementById('weatherInfo').classList.remove('active');
            document.getElementById('error').classList.remove('active');
            document.getElementById('apiNote').classList.remove('active');

            try {
                const response = await fetch('/weather?city=' + encodeURIComponent(city));
                const data = await response.json();

                document.getElementById('loading').classList.remove('active');

                if (data.error) {
                    showError(data.error);
                    if (data.note) {
                        document.getElementById('apiNote').textContent = data.note;
                        document.getElementById('apiNote').classList.add('active');
                    }
                    return;
                }

                document.getElementById('cityName').textContent = data.city;
                document.getElementById('country').textContent = data.country;
                document.getElementById('weatherIcon').textContent = weatherEmojis[data.main] || '🌡️';
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('description').textContent = data.description;
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('wind').textContent = data.wind_speed + ' m/s';
                document.getElementById('feelsLike').textContent = data.feels_like + '°C';

                document.getElementById('weatherInfo').classList.add('active');
            } catch (err) {
                document.getElementById('loading').classList.remove('active');
                showError('Failed to fetch weather data. Please try again.');
            }
        }

        function showError(msg) {
            document.getElementById('error').textContent = msg;
            document.getElementById('error').classList.add('active');
        }

        document.getElementById('cityInput').focus();
    </script>
</body>
</html>"""


def get_weather_from_wttr(city):
    """Fetch weather data from wttr.in (no API key required)."""
    try:
        url = "https://wttr.in/{}?format=j1".format(urllib.parse.quote(city))
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (WeatherApp/1.0)",
            "Accept": "application/json"
        })
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data["current_condition"][0]
        nearest_area = data["nearest_area"][0]

        city_name = nearest_area["areaName"][0]["value"]
        country = nearest_area["country"][0]["value"]
        region = nearest_area["region"][0]["value"]

        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind_speed_kmph = current["windspeedKmph"]
        wind_speed_ms = round(float(wind_speed_kmph) / 3.6, 1)
        description = current["weatherDesc"][0]["value"]

        weather_code = int(current.get("weatherCode", 0))
        if weather_code in (113,):
            main_weather = "Clear"
        elif weather_code in (116, 119, 122):
            main_weather = "Clouds"
        elif weather_code in (176, 263, 266, 281, 284, 293, 296, 299, 302, 305, 308, 311, 314, 353, 356, 359):
            main_weather = "Rain"
        elif weather_code in (200, 386, 389, 392, 395):
            main_weather = "Thunderstorm"
        elif weather_code in (179, 182, 185, 227, 230, 320, 323, 326, 329, 332, 335, 338, 350, 362, 365, 368, 371, 374, 377):
            main_weather = "Snow"
        elif weather_code in (143, 248, 260):
            main_weather = "Fog"
        else:
            main_weather = "Clouds"

        country_display = country
        if region:
            country_display = "{}, {}".format(region, country)

        return {
            "city": city_name,
            "country": country_display,
            "temperature": temp_c,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind_speed": wind_speed_ms,
            "description": description,
            "main": main_weather,
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": "City '{}' not found. Please check the spelling and try again.".format(city)}
        return {"error": "Weather service returned an error (HTTP {}). Please try again later.".format(e.code)}
    except urllib.error.URLError:
        return {"error": "Could not connect to weather service. Please check your internet connection."}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"error": "City '{}' not found or unexpected response from weather service.".format(city)}
    except Exception as e:
        return {"error": "An unexpected error occurred: {}".format(str(e))}


class WeatherHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path.startswith("/weather?"):
            query_string = self.path.split("?", 1)[1]
            params = urllib.parse.parse_qs(query_string)
            city = params.get("city", [""])[0]

            if not city:
                result = {"error": "Please provide a city name."}
            else:
                result = get_weather_from_wttr(city)

            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        print("[{}] {}".format(self.log_date_time_string(), format % args))


def main():
    print("=" * 50)
    print("  Weather App")
    print("=" * 50)
    print()
    print("  Using wttr.in API (no API key required!)")
    print()
    print("  Starting server on http://localhost:{}".format(PORT))
    print("  Open this URL in your browser.")
    print("  Press Ctrl+C to stop the server.")
    print()
    print("=" * 50)

    with socketserver.TCPServer(("", PORT), WeatherHandler) as httpd:
        httpd.allow_reuse_address = True
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
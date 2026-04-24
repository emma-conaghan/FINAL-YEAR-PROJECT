import http.client
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

API_KEY = "demo"
WTTR_BASE = "wttr.in"

HTML_TEMPLATE = """<!DOCTYPE html>
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
            padding: 20px;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2em;
        }
        .search-form {
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
        .weather-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
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
            margin-bottom: 20px;
            text-transform: capitalize;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .detail-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
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
            text-align: center;
            padding: 20px;
            font-size: 1.1em;
        }
        .weather-icon {
            font-size: 4em;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ Weather App</h1>
        <form class="search-form" action="/" method="GET">
            <input type="text" name="city" placeholder="Enter city name (e.g., London, Tokyo, New York)" value="{city_value}" required>
            <button type="submit">Search</button>
        </form>
        {weather_content}
    </div>
</body>
</html>"""

WEATHER_CARD_TEMPLATE = """
<div class="weather-card">
    <div class="city-name">{city}</div>
    <div class="weather-icon">{icon}</div>
    <div class="temperature">{temp}°C</div>
    <div class="description">{description}</div>
    <div class="details">
        <div class="detail-item">
            <div class="detail-label">Feels Like</div>
            <div class="detail-value">{feels_like}°C</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Humidity</div>
            <div class="detail-value">{humidity}%</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Wind Speed</div>
            <div class="detail-value">{wind} km/h</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Visibility</div>
            <div class="detail-value">{visibility} km</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Pressure</div>
            <div class="detail-value">{pressure} hPa</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">UV Index</div>
            <div class="detail-value">{uv}</div>
        </div>
    </div>
</div>
"""

ERROR_TEMPLATE = '<div class="error">❌ {message}</div>'

WEATHER_ICONS = {
    "Sunny": "☀️",
    "Clear": "🌙",
    "Partly cloudy": "⛅",
    "Cloudy": "☁️",
    "Overcast": "☁️",
    "Mist": "🌫️",
    "Fog": "🌫️",
    "Patchy rain possible": "🌦️",
    "Patchy rain nearby": "🌦️",
    "Light rain": "🌧️",
    "Moderate rain": "🌧️",
    "Heavy rain": "🌧️",
    "Light drizzle": "🌦️",
    "Thundery outbreaks possible": "⛈️",
    "Thunder": "⛈️",
    "Snow": "🌨️",
    "Light snow": "🌨️",
    "Heavy snow": "❄️",
    "Blizzard": "🌨️",
    "Sleet": "🌨️",
}


def get_weather_icon(description):
    for key, icon in WEATHER_ICONS.items():
        if key.lower() in description.lower():
            return icon
    return "🌡️"


def fetch_weather(city):
    try:
        encoded_city = urllib.parse.quote(city)
        conn = http.client.HTTPSConnection(WTTR_BASE, timeout=10)
        conn.request("GET", f"/{encoded_city}?format=j1", headers={"User-Agent": "WeatherApp/1.0"})
        response = conn.getresponse()

        if response.status != 200:
            return None, f"Could not fetch weather data. HTTP status: {response.status}"

        data = response.read().decode("utf-8")
        conn.close()

        weather_data = json.loads(data)

        if "current_condition" not in weather_data or not weather_data["current_condition"]:
            return None, f"No weather data found for '{city}'. Please check the city name."

        current = weather_data["current_condition"][0]

        nearest_area = weather_data.get("nearest_area", [{}])[0]
        area_name = nearest_area.get("areaName", [{}])[0].get("value", city) if nearest_area.get("areaName") else city
        country = nearest_area.get("country", [{}])[0].get("value", "") if nearest_area.get("country") else ""
        region = nearest_area.get("region", [{}])[0].get("value", "") if nearest_area.get("region") else ""

        location_parts = [area_name]
        if region and region != area_name:
            location_parts.append(region)
        if country:
            location_parts.append(country)
        location_str = ", ".join(location_parts)

        description = current.get("weatherDesc", [{}])[0].get("value", "Unknown")

        result = {
            "city": location_str,
            "temp": current.get("temp_C", "N/A"),
            "feels_like": current.get("FeelsLikeC", "N/A"),
            "humidity": current.get("humidity", "N/A"),
            "wind": current.get("windspeedKmph", "N/A"),
            "description": description,
            "visibility": current.get("visibility", "N/A"),
            "pressure": current.get("pressure", "N/A"),
            "uv": current.get("uvIndex", "N/A"),
            "icon": get_weather_icon(description),
        }

        return result, None

    except json.JSONDecodeError:
        return None, f"Could not parse weather data for '{city}'. The city may not exist."
    except http.client.HTTPException as e:
        return None, f"Network error: {str(e)}"
    except OSError as e:
        return None, f"Connection error: {str(e)}. Please check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def escape_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#x27;")


class WeatherHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        city = params.get("city", [""])[0].strip()
        weather_content = ""
        city_value = escape_html(city)

        if city:
            weather_data, error = fetch_weather(city)
            if error:
                weather_content = ERROR_TEMPLATE.format(message=escape_html(error))
            else:
                weather_content = WEATHER_CARD_TEMPLATE.format(
                    city=escape_html(weather_data["city"]),
                    icon=weather_data["icon"],
                    temp=escape_html(str(weather_data["temp"])),
                    description=escape_html(weather_data["description"]),
                    feels_like=escape_html(str(weather_data["feels_like"])),
                    humidity=escape_html(str(weather_data["humidity"])),
                    wind=escape_html(str(weather_data["wind"])),
                    visibility=escape_html(str(weather_data["visibility"])),
                    pressure=escape_html(str(weather_data["pressure"])),
                    uv=escape_html(str(weather_data["uv"])),
                )

        html = HTML_TEMPLATE.format(city_value=city_value, weather_content=weather_content)

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0] if args else ''}")


def main():
    host = "localhost"
    port = 8000

    server = HTTPServer((host, port), WeatherHandler)
    print("=" * 50)
    print("  🌤️  Weather App")
    print("=" * 50)
    print(f"  Server running at: http://{host}:{port}")
    print(f"  Using weather data from: wttr.in")
    print("  Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()
        print("Server stopped. Goodbye! 👋")


if __name__ == "__main__":
    main()
import urllib.request
import urllib.parse
import json
import sys

API_KEY = "demo"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

DEMO_DATA = {
    "london": {
        "name": "London",
        "sys": {"country": "GB"},
        "main": {"temp": 15.2, "feels_like": 13.8, "humidity": 76, "pressure": 1013},
        "weather": [{"description": "partly cloudy", "main": "Clouds"}],
        "wind": {"speed": 5.1},
        "visibility": 10000,
    },
    "new york": {
        "name": "New York",
        "sys": {"country": "US"},
        "main": {"temp": 22.4, "feels_like": 21.0, "humidity": 60, "pressure": 1008},
        "weather": [{"description": "clear sky", "main": "Clear"}],
        "wind": {"speed": 3.6},
        "visibility": 10000,
    },
    "tokyo": {
        "name": "Tokyo",
        "sys": {"country": "JP"},
        "main": {"temp": 28.1, "feels_like": 30.5, "humidity": 85, "pressure": 1005},
        "weather": [{"description": "light rain", "main": "Rain"}],
        "wind": {"speed": 2.1},
        "visibility": 8000,
    },
    "paris": {
        "name": "Paris",
        "sys": {"country": "FR"},
        "main": {"temp": 18.7, "feels_like": 17.5, "humidity": 65, "pressure": 1015},
        "weather": [{"description": "overcast clouds", "main": "Clouds"}],
        "wind": {"speed": 4.2},
        "visibility": 10000,
    },
    "sydney": {
        "name": "Sydney",
        "sys": {"country": "AU"},
        "main": {"temp": 20.3, "feels_like": 19.8, "humidity": 70, "pressure": 1018},
        "weather": [{"description": "sunny", "main": "Clear"}],
        "wind": {"speed": 6.0},
        "visibility": 10000,
    },
}

WEATHER_ICONS = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧️",
    "Drizzle": "🌦️",
    "Thunderstorm": "⛈️",
    "Snow": "❄️",
    "Mist": "🌫️",
    "Fog": "🌫️",
    "Haze": "🌫️",
    "Smoke": "🌫️",
    "Dust": "🌫️",
    "Sand": "🌫️",
    "Ash": "🌫️",
    "Squall": "💨",
    "Tornado": "🌪️",
}


def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32


def get_visibility_label(visibility_meters):
    if visibility_meters >= 10000:
        return "Excellent (10+ km)"
    elif visibility_meters >= 5000:
        return f"Good ({visibility_meters / 1000:.1f} km)"
    elif visibility_meters >= 1000:
        return f"Moderate ({visibility_meters / 1000:.1f} km)"
    else:
        return f"Poor ({visibility_meters} m)"


def get_wind_description(speed_ms):
    if speed_ms < 0.5:
        return "Calm"
    elif speed_ms < 1.6:
        return "Light air"
    elif speed_ms < 3.4:
        return "Light breeze"
    elif speed_ms < 5.5:
        return "Gentle breeze"
    elif speed_ms < 8.0:
        return "Moderate breeze"
    elif speed_ms < 10.8:
        return "Fresh breeze"
    elif speed_ms < 13.9:
        return "Strong breeze"
    else:
        return "High wind"


def display_weather(data):
    city_name = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "Unknown")
    main_data = data.get("main", {})
    weather_list = data.get("weather", [{}])
    weather_desc = weather_list[0].get("description", "Unknown").title()
    weather_main = weather_list[0].get("main", "")
    wind_data = data.get("wind", {})
    visibility = data.get("visibility", None)

    temp_c = main_data.get("temp", 0)
    feels_like_c = main_data.get("feels_like", 0)
    humidity = main_data.get("humidity", 0)
    pressure = main_data.get("pressure", 0)
    wind_speed = wind_data.get("speed", 0)

    temp_f = celsius_to_fahrenheit(temp_c)
    feels_like_f = celsius_to_fahrenheit(feels_like_c)
    wind_kmh = wind_speed * 3.6

    icon = WEATHER_ICONS.get(weather_main, "🌡️")
    wind_desc = get_wind_description(wind_speed)

    print()
    print("=" * 50)
    print(f"  {icon}  Weather in {city_name}, {country}")
    print("=" * 50)
    print(f"  Condition   : {weather_desc}")
    print(f"  Temperature : {temp_c:.1f}°C / {temp_f:.1f}°F")
    print(f"  Feels Like  : {feels_like_c:.1f}°C / {feels_like_f:.1f}°F")
    print(f"  Humidity    : {humidity}%")
    print(f"  Pressure    : {pressure} hPa")
    print(f"  Wind        : {wind_speed:.1f} m/s ({wind_kmh:.1f} km/h) - {wind_desc}")
    if visibility is not None:
        print(f"  Visibility  : {get_visibility_label(visibility)}")
    print("=" * 50)
    print()


def fetch_weather_real(city, api_key):
    params = urllib.parse.urlencode({
        "q": city,
        "appid": api_key,
        "units": "metric",
    })
    url = f"{BASE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw), None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "Invalid API key. Please check your key at https://openweathermap.org"
        elif e.code == 404:
            return None, f"City '{city}' not found."
        else:
            return None, f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"Network error: {e.reason}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def fetch_weather_demo(city):
    city_lower = city.strip().lower()
    if city_lower in DEMO_DATA:
        return DEMO_DATA[city_lower], None
    return None, f"City '{city}' not found in demo data."


def print_demo_notice():
    print()
    print("┌─────────────────────────────────────────────────┐")
    print("│              DEMO MODE ACTIVE                   │")
    print("│                                                 │")
    print("│  No real API key configured.                    │")
    print("│  Using built-in sample data for:               │")
    print("│    • London  • New York  • Tokyo               │")
    print("│    • Paris   • Sydney                          │")
    print("│                                                 │")
    print("│  To use real weather data:                      │")
    print("│  1. Get a free key at openweathermap.org        │")
    print("│  2. Set API_KEY at the top of app.py            │")
    print("└─────────────────────────────────────────────────┘")


def print_banner():
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║        🌤  Weather App  🌤            ║")
    print("  ║   Real-time weather information       ║")
    print("  ╚══════════════════════════════════════╝")


def main():
    print_banner()

    use_demo = API_KEY == "demo"

    if use_demo:
        print_demo_notice()

    print()
    print("Type a city name to get weather info.")
    print("Type 'quit' or 'exit' to close the app.")
    print()

    while True:
        try:
            city = input("Enter city name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not city:
            print("Please enter a city name.")
            continue

        if city.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            sys.exit(0)

        print(f"\nFetching weather for '{city}'...")

        if use_demo:
            data, error = fetch_weather_demo(city)
        else:
            data, error = fetch_weather_real(city, API_KEY)

        if error:
            print(f"  ⚠️  Error: {error}")
            if use_demo:
                available = ", ".join(k.title() for k in DEMO_DATA.keys())
                print(f"  Available cities in demo mode: {available}")
        else:
            display_weather(data)

        print()


if __name__ == "__main__":
    main()
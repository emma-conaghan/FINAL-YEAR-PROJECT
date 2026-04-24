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
        "main": {"temp": 15.2, "feels_like": 13.8, "humidity": 72, "pressure": 1013},
        "weather": [{"description": "light rain", "main": "Rain"}],
        "wind": {"speed": 5.1},
        "visibility": 10000,
    },
    "new york": {
        "name": "New York",
        "sys": {"country": "US"},
        "main": {"temp": 22.5, "feels_like": 21.0, "humidity": 60, "pressure": 1015},
        "weather": [{"description": "clear sky", "main": "Clear"}],
        "wind": {"speed": 3.6},
        "visibility": 16000,
    },
    "tokyo": {
        "name": "Tokyo",
        "sys": {"country": "JP"},
        "main": {"temp": 28.0, "feels_like": 30.2, "humidity": 80, "pressure": 1008},
        "weather": [{"description": "few clouds", "main": "Clouds"}],
        "wind": {"speed": 2.1},
        "visibility": 12000,
    },
    "paris": {
        "name": "Paris",
        "sys": {"country": "FR"},
        "main": {"temp": 18.0, "feels_like": 17.0, "humidity": 65, "pressure": 1012},
        "weather": [{"description": "overcast clouds", "main": "Clouds"}],
        "wind": {"speed": 4.0},
        "visibility": 10000,
    },
    "sydney": {
        "name": "Sydney",
        "sys": {"country": "AU"},
        "main": {"temp": 20.0, "feels_like": 19.5, "humidity": 68, "pressure": 1016},
        "weather": [{"description": "sunny", "main": "Clear"}],
        "wind": {"speed": 6.0},
        "visibility": 15000,
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
    "Squall": "🌬️",
    "Tornado": "🌪️",
}


def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32


def get_weather_icon(condition):
    return WEATHER_ICONS.get(condition, "🌡️")


def display_weather(data):
    city = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "")
    temp_c = data.get("main", {}).get("temp", 0)
    feels_like_c = data.get("main", {}).get("feels_like", 0)
    humidity = data.get("main", {}).get("humidity", 0)
    pressure = data.get("main", {}).get("pressure", 0)
    wind_speed = data.get("wind", {}).get("speed", 0)
    visibility_m = data.get("visibility", 0)
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "N/A").capitalize()
    condition = weather_list[0].get("main", "")

    temp_f = celsius_to_fahrenheit(temp_c)
    feels_like_f = celsius_to_fahrenheit(feels_like_c)
    visibility_km = visibility_m / 1000 if visibility_m else 0
    icon = get_weather_icon(condition)

    print("\n" + "=" * 50)
    print(f"  {icon}  Weather for {city}, {country}")
    print("=" * 50)
    print(f"  Condition    : {description}")
    print(f"  Temperature  : {temp_c:.1f}°C / {temp_f:.1f}°F")
    print(f"  Feels Like   : {feels_like_c:.1f}°C / {feels_like_f:.1f}°F")
    print(f"  Humidity     : {humidity}%")
    print(f"  Pressure     : {pressure} hPa")
    print(f"  Wind Speed   : {wind_speed} m/s")
    print(f"  Visibility   : {visibility_km:.1f} km")
    print("=" * 50 + "\n")


def fetch_weather_from_api(city, api_key):
    params = urllib.parse.urlencode({
        "q": city,
        "appid": api_key,
        "units": "metric",
    })
    url = f"{BASE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw = response.read()
            data = json.loads(raw)
            return data, None
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            err_data = json.loads(raw)
            msg = err_data.get("message", str(e))
        except Exception:
            msg = str(e)
        return None, f"HTTP Error {e.code}: {msg}"
    except urllib.error.URLError as e:
        return None, f"Network error: {e.reason}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def fetch_weather_demo(city):
    key = city.strip().lower()
    if key in DEMO_DATA:
        return DEMO_DATA[key], None
    return None, f"City '{city}' not found in demo data."


def get_api_key():
    print("\nThis app uses the OpenWeatherMap API.")
    print("You can get a free API key at: https://openweathermap.org/api")
    print("(It may take a few minutes after signup before the key works.)\n")
    print("Options:")
    print("  1. Enter your own API key")
    print("  2. Use demo mode (no API key needed, limited cities)")
    print()
    choice = input("Choose an option (1 or 2): ").strip()
    if choice == "1":
        key = input("Enter your OpenWeatherMap API key: ").strip()
        if not key:
            print("No key entered. Switching to demo mode.")
            return None
        return key
    return None


def print_demo_cities():
    print("\nAvailable cities in demo mode:")
    for city in DEMO_DATA:
        print(f"  - {city.title()}")
    print()


def main():
    print("=" * 50)
    print("       🌤️  Weather Information App  🌤️")
    print("=" * 50)

    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        demo_mode = False
        print(f"Using API key from command line argument.")
    else:
        api_key = get_api_key()
        demo_mode = api_key is None

    if demo_mode:
        print("\n[DEMO MODE] No real API calls will be made.")
        print_demo_cities()

    while True:
        city = input("Enter city name (or 'quit' to exit): ").strip()
        if city.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break
        if not city:
            print("Please enter a valid city name.\n")
            continue

        print(f"\nFetching weather for '{city}'...")

        if demo_mode:
            data, error = fetch_weather_demo(city)
        else:
            data, error = fetch_weather_from_api(city, api_key)

        if error:
            print(f"❌ Error: {error}")
            if demo_mode:
                print_demo_cities()
            else:
                print("Please check the city name and try again.\n")
        else:
            display_weather(data)

        again = input("Search another city? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("Goodbye! 👋")
            break


if __name__ == "__main__":
    main()
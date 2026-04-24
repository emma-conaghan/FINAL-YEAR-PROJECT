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
        "weather": [{"description": "light rain", "main": "Rain"}],
        "main": {
            "temp": 280.15,
            "feels_like": 278.0,
            "humidity": 81,
            "pressure": 1012,
            "temp_min": 279.15,
            "temp_max": 281.15,
        },
        "wind": {"speed": 5.1, "deg": 200},
        "visibility": 10000,
        "clouds": {"all": 75},
    },
    "new york": {
        "name": "New York",
        "sys": {"country": "US"},
        "weather": [{"description": "clear sky", "main": "Clear"}],
        "main": {
            "temp": 295.37,
            "feels_like": 295.11,
            "humidity": 53,
            "pressure": 1018,
            "temp_min": 293.71,
            "temp_max": 297.04,
        },
        "wind": {"speed": 3.6, "deg": 150},
        "visibility": 10000,
        "clouds": {"all": 0},
    },
    "tokyo": {
        "name": "Tokyo",
        "sys": {"country": "JP"},
        "weather": [{"description": "few clouds", "main": "Clouds"}],
        "main": {
            "temp": 302.15,
            "feels_like": 305.0,
            "humidity": 68,
            "pressure": 1008,
            "temp_min": 300.15,
            "temp_max": 304.15,
        },
        "wind": {"speed": 2.5, "deg": 90},
        "visibility": 10000,
        "clouds": {"all": 20},
    },
    "sydney": {
        "name": "Sydney",
        "sys": {"country": "AU"},
        "weather": [{"description": "sunny", "main": "Clear"}],
        "main": {
            "temp": 298.15,
            "feels_like": 297.5,
            "humidity": 60,
            "pressure": 1015,
            "temp_min": 296.15,
            "temp_max": 300.15,
        },
        "wind": {"speed": 4.0, "deg": 270},
        "visibility": 10000,
        "clouds": {"all": 5},
    },
    "paris": {
        "name": "Paris",
        "sys": {"country": "FR"},
        "weather": [{"description": "overcast clouds", "main": "Clouds"}],
        "main": {
            "temp": 283.15,
            "feels_like": 281.5,
            "humidity": 76,
            "pressure": 1010,
            "temp_min": 282.15,
            "temp_max": 284.15,
        },
        "wind": {"speed": 3.0, "deg": 180},
        "visibility": 9000,
        "clouds": {"all": 90},
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
    "Dust": "🌪️",
    "Sand": "🌪️",
    "Ash": "🌋",
    "Squall": "💨",
    "Tornado": "🌪️",
}


def kelvin_to_celsius(kelvin):
    return kelvin - 273.15


def kelvin_to_fahrenheit(kelvin):
    return (kelvin - 273.15) * 9 / 5 + 32


def get_weather_icon(weather_main):
    return WEATHER_ICONS.get(weather_main, "🌡️")


def fetch_weather_live(city):
    params = urllib.parse.urlencode({"q": city, "appid": API_KEY})
    url = f"{BASE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read()
            return json.loads(data), None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, "City not found. Please check the city name and try again."
        elif e.code == 401:
            return None, "Invalid API key. Please set a valid API key."
        elif e.code == 429:
            return None, "Too many requests. Please wait a moment and try again."
        else:
            return None, f"HTTP error occurred: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return None, f"Network error: {e.reason}"
    except json.JSONDecodeError:
        return None, "Failed to parse weather data."
    except Exception as e:
        return None, f"Unexpected error: {e}"


def fetch_weather_demo(city):
    city_lower = city.lower().strip()
    if city_lower in DEMO_DATA:
        return DEMO_DATA[city_lower], None
    else:
        available = ", ".join(c.title() for c in DEMO_DATA.keys())
        return None, (
            f"City '{city}' not found in demo data.\n"
            f"Available demo cities: {available}"
        )


def display_weather(data):
    city_name = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "Unknown")
    weather_list = data.get("weather", [{}])
    weather_main = weather_list[0].get("main", "Unknown") if weather_list else "Unknown"
    weather_desc = weather_list[0].get("description", "Unknown") if weather_list else "Unknown"
    main_data = data.get("main", {})
    temp_k = main_data.get("temp", 0)
    feels_like_k = main_data.get("feels_like", 0)
    temp_min_k = main_data.get("temp_min", 0)
    temp_max_k = main_data.get("temp_max", 0)
    humidity = main_data.get("humidity", 0)
    pressure = main_data.get("pressure", 0)
    wind_data = data.get("wind", {})
    wind_speed = wind_data.get("speed", 0)
    wind_deg = wind_data.get("deg", 0)
    visibility = data.get("visibility", 0)
    clouds = data.get("clouds", {}).get("all", 0)

    temp_c = kelvin_to_celsius(temp_k)
    temp_f = kelvin_to_fahrenheit(temp_k)
    feels_c = kelvin_to_celsius(feels_like_k)
    feels_f = kelvin_to_fahrenheit(feels_like_k)
    temp_min_c = kelvin_to_celsius(temp_min_k)
    temp_max_c = kelvin_to_celsius(temp_max_k)
    temp_min_f = kelvin_to_fahrenheit(temp_min_k)
    temp_max_f = kelvin_to_fahrenheit(temp_max_k)

    icon = get_weather_icon(weather_main)
    wind_dir = get_wind_direction(wind_deg)
    vis_km = visibility / 1000 if visibility else 0

    width = 55
    separator = "=" * width
    thin_sep = "-" * width

    print()
    print(separator)
    print(f"  {icon}  Weather Report: {city_name}, {country}")
    print(separator)
    print(f"  Condition     : {weather_desc.title()}")
    print(thin_sep)
    print(f"  Temperature   : {temp_c:.1f}°C  /  {temp_f:.1f}°F")
    print(f"  Feels Like    : {feels_c:.1f}°C  /  {feels_f:.1f}°F")
    print(f"  Min / Max     : {temp_min_c:.1f}°C  /  {temp_max_c:.1f}°C")
    print(f"                  ({temp_min_f:.1f}°F  /  {temp_max_f:.1f}°F)")
    print(thin_sep)
    print(f"  Humidity      : {humidity}%")
    print(f"  Pressure      : {pressure} hPa")
    print(f"  Wind          : {wind_speed} m/s {wind_dir} ({wind_deg}°)")
    print(f"  Visibility    : {vis_km:.1f} km")
    print(f"  Cloud Cover   : {clouds}%")
    print(separator)
    print()


def get_wind_direction(degrees):
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW",
    ]
    index = round(degrees / 22.5) % 16
    return directions[index]


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║           🌍  Weather Information App  🌍            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def print_mode_info(use_live):
    if use_live:
        print("  Mode: LIVE (OpenWeatherMap API)")
        print(f"  API Key: {API_KEY[:8]}..." if len(API_KEY) > 8 else f"  API Key: {API_KEY}")
        print()
        print("  NOTE: To use live data, replace the API_KEY variable")
        print("  at the top of this file with your own key from:")
        print("  https://openweathermap.org/api  (free registration)")
        print()
    else:
        print("  Mode: DEMO (offline, no API key required)")
        print()
        available = ", ".join(c.title() for c in DEMO_DATA.keys())
        print(f"  Available cities: {available}")
        print()
        print("  To use live data for any city, edit app.py and:")
        print("  1. Set API_KEY to your OpenWeatherMap API key")
        print("  2. Set use_live = True in the main() function")
        print()


def main():
    use_live = False

    print_banner()
    print_mode_info(use_live)

    while True:
        try:
            city = input("  Enter city name (or 'quit' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye! 👋\n")
            sys.exit(0)

        if not city:
            print("  Please enter a valid city name.\n")
            continue

        if city.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋\n")
            sys.exit(0)

        print(f"\n  Fetching weather data for '{city}'...\n")

        if use_live:
            data, error = fetch_weather_live(city)
        else:
            data, error = fetch_weather_demo(city)

        if error:
            print(f"  ❌ Error: {error}\n")
        else:
            display_weather(data)

        try:
            again = input("  Search for another city? (yes/no): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye! 👋\n")
            sys.exit(0)

        if again not in ("yes", "y"):
            print("\n  Goodbye! 👋\n")
            break


if __name__ == "__main__":
    main()
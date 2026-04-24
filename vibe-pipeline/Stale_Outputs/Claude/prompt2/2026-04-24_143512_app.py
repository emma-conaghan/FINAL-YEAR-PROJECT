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
        "weather": [{"description": "overcast clouds", "main": "Clouds"}],
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
        "main": {"temp": 28.1, "feels_like": 30.5, "humidity": 80, "pressure": 1005},
        "weather": [{"description": "light rain", "main": "Rain"}],
        "wind": {"speed": 2.1},
        "visibility": 8000,
    },
    "paris": {
        "name": "Paris",
        "sys": {"country": "FR"},
        "main": {"temp": 18.7, "feels_like": 17.9, "humidity": 65, "pressure": 1015},
        "weather": [{"description": "few clouds", "main": "Clouds"}],
        "wind": {"speed": 4.2},
        "visibility": 10000,
    },
    "sydney": {
        "name": "Sydney",
        "sys": {"country": "AU"},
        "main": {"temp": 12.3, "feels_like": 11.0, "humidity": 55, "pressure": 1020},
        "weather": [{"description": "sunny", "main": "Clear"}],
        "wind": {"speed": 6.7},
        "visibility": 10000,
    },
    "dubai": {
        "name": "Dubai",
        "sys": {"country": "AE"},
        "main": {"temp": 38.5, "feels_like": 42.0, "humidity": 45, "pressure": 1001},
        "weather": [{"description": "hot and sunny", "main": "Clear"}],
        "wind": {"speed": 1.5},
        "visibility": 10000,
    },
    "moscow": {
        "name": "Moscow",
        "sys": {"country": "RU"},
        "main": {"temp": -5.2, "feels_like": -9.8, "humidity": 85, "pressure": 1025},
        "weather": [{"description": "light snow", "main": "Snow"}],
        "wind": {"speed": 3.3},
        "visibility": 5000,
    },
    "cairo": {
        "name": "Cairo",
        "sys": {"country": "EG"},
        "main": {"temp": 30.1, "feels_like": 29.5, "humidity": 30, "pressure": 1010},
        "weather": [{"description": "clear sky", "main": "Clear"}],
        "wind": {"speed": 4.0},
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


def get_weather_icon(main_condition):
    return WEATHER_ICONS.get(main_condition, "🌡️")


def fetch_weather_from_api(city, api_key):
    params = urllib.parse.urlencode({"q": city, "appid": api_key, "units": "metric"})
    url = f"{BASE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read()
            return json.loads(data), None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, "City not found. Please check the city name and try again."
        elif e.code == 401:
            return None, "Invalid API key. Please check your API key."
        elif e.code == 429:
            return None, "API rate limit exceeded. Please try again later."
        else:
            return None, f"HTTP error occurred: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return None, f"Network error: {e.reason}"
    except json.JSONDecodeError:
        return None, "Failed to parse weather data from API."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def get_demo_weather(city):
    city_lower = city.strip().lower()
    if city_lower in DEMO_DATA:
        return DEMO_DATA[city_lower], None
    return None, f"City '{city}' not found in demo data.\nAvailable demo cities: {', '.join(c.title() for c in DEMO_DATA.keys())}"


def display_weather(data):
    name = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "??")
    main = data.get("main", {})
    weather_list = data.get("weather", [{}])
    weather_desc = weather_list[0].get("description", "N/A").capitalize()
    weather_main = weather_list[0].get("main", "")
    wind = data.get("wind", {})
    visibility = data.get("visibility", None)

    temp_c = main.get("temp", 0)
    feels_like_c = main.get("feels_like", 0)
    humidity = main.get("humidity", 0)
    pressure = main.get("pressure", 0)
    wind_speed = wind.get("speed", 0)

    temp_f = celsius_to_fahrenheit(temp_c)
    feels_like_f = celsius_to_fahrenheit(feels_like_c)
    icon = get_weather_icon(weather_main)

    width = 50
    border = "=" * width

    print()
    print(border)
    print(f"  {icon}  Weather in {name}, {country}")
    print(border)
    print(f"  Condition    : {weather_desc}")
    print(f"  Temperature  : {temp_c:.1f}°C  /  {temp_f:.1f}°F")
    print(f"  Feels Like   : {feels_like_c:.1f}°C  /  {feels_like_f:.1f}°F")
    print(f"  Humidity     : {humidity}%")
    print(f"  Pressure     : {pressure} hPa")
    print(f"  Wind Speed   : {wind_speed} m/s  ({wind_speed * 3.6:.1f} km/h)")
    if visibility is not None:
        vis_km = visibility / 1000
        print(f"  Visibility   : {vis_km:.1f} km")
    print(border)
    print()


def get_temperature_advice(temp_c):
    if temp_c < -10:
        return "🥶 Extremely cold! Bundle up with heavy winter clothing."
    elif temp_c < 0:
        return "🧥 Very cold. Wear a heavy coat, hat, and gloves."
    elif temp_c < 10:
        return "🧤 Cold. A warm jacket is recommended."
    elif temp_c < 18:
        return "🍂 Cool. A light jacket or sweater would be comfortable."
    elif temp_c < 25:
        return "😊 Pleasant temperature. Enjoy the weather!"
    elif temp_c < 32:
        return "☀️ Warm. Light clothing recommended."
    elif temp_c < 38:
        return "🌡️ Hot! Stay hydrated and seek shade."
    else:
        return "🔥 Extremely hot! Avoid prolonged sun exposure and drink plenty of water."


def show_advice(data):
    main = data.get("main", {})
    weather_list = data.get("weather", [{}])
    weather_main = weather_list[0].get("main", "")
    temp_c = main.get("temp", 0)
    humidity = main.get("humidity", 0)

    advice = []
    advice.append(get_temperature_advice(temp_c))

    if weather_main in ("Rain", "Drizzle", "Thunderstorm"):
        advice.append("☂️  Don't forget your umbrella!")
    if weather_main == "Snow":
        advice.append("🌨️  Roads may be slippery. Drive carefully.")
    if weather_main in ("Fog", "Mist", "Haze"):
        advice.append("👓 Low visibility. Take extra care if driving.")
    if humidity > 80:
        advice.append("💧 High humidity. It may feel warmer than it is.")
    if weather_main == "Thunderstorm":
        advice.append("⚡ Thunderstorm alert! Stay indoors if possible.")

    print("  💡 Weather Advice:")
    for tip in advice:
        print(f"     • {tip}")
    print()


def run_app():
    print("=" * 50)
    print("       🌤️  Weather Information App")
    print("=" * 50)
    print()

    use_demo = True
    custom_key = None

    if API_KEY == "demo":
        print("  ℹ️  Running in DEMO MODE with sample weather data.")
        print("  To use real data, set your OpenWeatherMap API key")
        print("  by replacing 'demo' in the API_KEY variable at")
        print("  the top of app.py with your actual key.")
        print()
        print("  Get a free API key at: https://openweathermap.org/api")
        print()
    else:
        use_demo = False
        custom_key = API_KEY
        print("  ✅ Using OpenWeatherMap API with provided key.")
        print()

    while True:
        if use_demo:
            available = ", ".join(c.title() for c in DEMO_DATA.keys())
            print(f"  Available cities (demo): {available}")
            print()

        try:
            city = input("  Enter city name (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye! 👋\n")
            sys.exit(0)

        if not city:
            print("  ⚠️  Please enter a city name.\n")
            continue

        if city.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋\n")
            break

        print(f"\n  🔍 Fetching weather data for '{city}'...\n")

        if use_demo:
            weather_data, error = get_demo_weather(city)
        else:
            weather_data, error = fetch_weather_from_api(city, custom_key)

        if error:
            print(f"  ❌ Error: {error}\n")
        else:
            display_weather(weather_data)
            show_advice(weather_data)

        try:
            again = input("  Search another city? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye! 👋\n")
            sys.exit(0)

        if again not in ("yes", "y"):
            print("\n  Goodbye! 👋\n")
            break

        print()


if __name__ == "__main__":
    run_app()
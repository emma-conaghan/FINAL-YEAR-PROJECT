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
        "weather": [{"description": "overcast clouds", "main": "Clouds"}],
        "main": {
            "temp": 15.2,
            "feels_like": 14.1,
            "temp_min": 13.0,
            "temp_max": 17.0,
            "humidity": 76,
            "pressure": 1012,
        },
        "wind": {"speed": 5.1, "deg": 220},
        "visibility": 10000,
    },
    "new york": {
        "name": "New York",
        "sys": {"country": "US"},
        "weather": [{"description": "clear sky", "main": "Clear"}],
        "main": {
            "temp": 22.5,
            "feels_like": 21.8,
            "temp_min": 19.0,
            "temp_max": 25.0,
            "humidity": 55,
            "pressure": 1018,
        },
        "wind": {"speed": 3.6, "deg": 180},
        "visibility": 16000,
    },
    "tokyo": {
        "name": "Tokyo",
        "sys": {"country": "JP"},
        "weather": [{"description": "light rain", "main": "Rain"}],
        "main": {
            "temp": 19.0,
            "feels_like": 18.5,
            "temp_min": 17.0,
            "temp_max": 21.0,
            "humidity": 82,
            "pressure": 1008,
        },
        "wind": {"speed": 4.2, "deg": 90},
        "visibility": 8000,
    },
    "sydney": {
        "name": "Sydney",
        "sys": {"country": "AU"},
        "weather": [{"description": "sunny", "main": "Clear"}],
        "main": {
            "temp": 24.0,
            "feels_like": 23.5,
            "temp_min": 20.0,
            "temp_max": 27.0,
            "humidity": 60,
            "pressure": 1020,
        },
        "wind": {"speed": 6.0, "deg": 270},
        "visibility": 20000,
    },
    "paris": {
        "name": "Paris",
        "sys": {"country": "FR"},
        "weather": [{"description": "partly cloudy", "main": "Clouds"}],
        "main": {
            "temp": 18.0,
            "feels_like": 17.2,
            "temp_min": 15.0,
            "temp_max": 20.0,
            "humidity": 68,
            "pressure": 1015,
        },
        "wind": {"speed": 4.5, "deg": 200},
        "visibility": 12000,
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


def fetch_weather_live(city, api_key):
    params = urllib.parse.urlencode({
        "q": city,
        "appid": api_key,
        "units": "metric",
    })
    url = f"{BASE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data, None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "Invalid API key. Please check your API key."
        elif e.code == 404:
            return None, f"City '{city}' not found."
        else:
            return None, f"HTTP error occurred: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return None, f"Network error: {e.reason}"
    except json.JSONDecodeError:
        return None, "Failed to parse weather data."
    except Exception as e:
        return None, f"Unexpected error: {e}"


def fetch_weather_demo(city):
    key = city.lower().strip()
    if key in DEMO_DATA:
        data = DEMO_DATA[key]
        converted = {
            "name": data["name"],
            "sys": data["sys"],
            "weather": data["weather"],
            "main": {
                "temp": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "temp_min": data["main"]["temp_min"],
                "temp_max": data["main"]["temp_max"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
            },
            "wind": data["wind"],
            "visibility": data["visibility"],
            "_demo": True,
        }
        return converted, None
    else:
        available = ", ".join(k.title() for k in DEMO_DATA.keys())
        return None, (
            f"City '{city}' not found in demo data.\n"
            f"Available demo cities: {available}\n"
            f"To use real data, set your OpenWeatherMap API key in the script."
        )


def display_weather(data):
    name = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "")
    weather_list = data.get("weather", [{}])
    weather_main = weather_list[0].get("main", "Unknown") if weather_list else "Unknown"
    weather_desc = weather_list[0].get("description", "Unknown") if weather_list else "Unknown"
    main = data.get("main", {})
    wind = data.get("wind", {})
    visibility = data.get("visibility", None)
    is_demo = data.get("_demo", False)

    icon = WEATHER_ICONS.get(weather_main, "🌡️")

    temp = main.get("temp", 0)
    feels_like = main.get("feels_like", 0)
    temp_min = main.get("temp_min", 0)
    temp_max = main.get("temp_max", 0)
    humidity = main.get("humidity", 0)
    pressure = main.get("pressure", 0)
    wind_speed = wind.get("speed", 0)
    wind_deg = wind.get("deg", 0)

    temp_f = temp * 9 / 5 + 32
    feels_like_f = feels_like * 9 / 5 + 32
    temp_min_f = temp_min * 9 / 5 + 32
    temp_max_f = temp_max * 9 / 5 + 32

    wind_directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    wind_dir_str = wind_directions[int((wind_deg + 11.25) / 22.5) % 16]

    vis_str = f"{visibility / 1000:.1f} km" if visibility is not None else "N/A"

    separator = "=" * 50
    print()
    print(separator)
    if is_demo:
        print("  ⚠️  DEMO MODE - Sample Data")
    print(f"  {icon}  Weather for {name}, {country}")
    print(separator)
    print(f"  Condition   : {weather_desc.title()}")
    print(f"  Temperature : {temp:.1f}°C / {temp_f:.1f}°F")
    print(f"  Feels Like  : {feels_like:.1f}°C / {feels_like_f:.1f}°F")
    print(f"  Min / Max   : {temp_min:.1f}°C ({temp_min_f:.1f}°F) / {temp_max:.1f}°C ({temp_max_f:.1f}°F)")
    print(f"  Humidity    : {humidity}%")
    print(f"  Pressure    : {pressure} hPa")
    print(f"  Wind        : {wind_speed} m/s {wind_dir_str} ({wind_deg}°)")
    print(f"  Visibility  : {vis_str}")
    print(separator)
    print()


def get_wind_description(speed):
    if speed < 0.5:
        return "Calm"
    elif speed < 1.5:
        return "Light air"
    elif speed < 3.3:
        return "Light breeze"
    elif speed < 5.5:
        return "Gentle breeze"
    elif speed < 7.9:
        return "Moderate breeze"
    elif speed < 10.7:
        return "Fresh breeze"
    elif speed < 13.8:
        return "Strong breeze"
    elif speed < 17.1:
        return "High wind"
    elif speed < 20.7:
        return "Gale"
    elif speed < 24.4:
        return "Strong gale"
    elif speed < 28.4:
        return "Storm"
    elif speed < 32.6:
        return "Violent storm"
    else:
        return "Hurricane force"


def print_banner():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║        🌤️  Weather Info App  🌤️           ║")
    print("║   Powered by OpenWeatherMap API          ║")
    print("╚══════════════════════════════════════════╝")
    print()


def print_help():
    print("Commands:")
    print("  <city name>  - Get weather for a city")
    print("  help         - Show this help message")
    print("  demo         - List available demo cities")
    print("  quit / exit  - Exit the application")
    print()


def main():
    print_banner()

    use_demo = API_KEY == "demo"

    if use_demo:
        print("ℹ️  Running in DEMO MODE with sample data.")
        print("   To use real weather data:")
        print("   1. Sign up at https://openweathermap.org/api")
        print("   2. Get your free API key")
        print("   3. Replace 'demo' with your key in the script (API_KEY variable)")
        print()
    else:
        print(f"ℹ️  Using OpenWeatherMap API with key: {API_KEY[:4]}{'*' * (len(API_KEY) - 4)}")
        print()

    print_help()

    while True:
        try:
            city_input = input("Enter city name (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            sys.exit(0)

        if not city_input:
            print("Please enter a city name.\n")
            continue

        lower_input = city_input.lower()

        if lower_input in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            sys.exit(0)

        if lower_input == "help":
            print_help()
            continue

        if lower_input == "demo":
            available = ", ".join(k.title() for k in DEMO_DATA.keys())
            print(f"Available demo cities: {available}\n")
            continue

        print(f"\n🔍 Fetching weather data for '{city_input}'...")

        if use_demo:
            data, error = fetch_weather_demo(city_input)
        else:
            data, error = fetch_weather_live(city_input, API_KEY)

        if error:
            print(f"\n❌ Error: {error}\n")
        else:
            display_weather(data)

            if data.get("wind"):
                wind_speed = data["wind"].get("speed", 0)
                wind_desc = get_wind_description(wind_speed)
                print(f"  💨 Wind Description: {wind_desc} ({wind_speed} m/s)\n")


if __name__ == "__main__":
    main()
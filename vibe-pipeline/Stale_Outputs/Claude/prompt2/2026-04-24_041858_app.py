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
        "main": {
            "temp": 15.2,
            "feels_like": 13.8,
            "humidity": 76,
            "pressure": 1012,
            "temp_min": 12.0,
            "temp_max": 17.5
        },
        "weather": [{"description": "overcast clouds", "main": "Clouds"}],
        "wind": {"speed": 4.1, "deg": 220},
        "visibility": 10000,
        "clouds": {"all": 90}
    },
    "new york": {
        "name": "New York",
        "sys": {"country": "US"},
        "main": {
            "temp": 22.4,
            "feels_like": 21.9,
            "humidity": 55,
            "pressure": 1018,
            "temp_min": 19.0,
            "temp_max": 25.0
        },
        "weather": [{"description": "clear sky", "main": "Clear"}],
        "wind": {"speed": 3.6, "deg": 180},
        "visibility": 16000,
        "clouds": {"all": 5}
    },
    "tokyo": {
        "name": "Tokyo",
        "sys": {"country": "JP"},
        "main": {
            "temp": 28.7,
            "feels_like": 30.1,
            "humidity": 80,
            "pressure": 1008,
            "temp_min": 25.0,
            "temp_max": 31.0
        },
        "weather": [{"description": "light rain", "main": "Rain"}],
        "wind": {"speed": 2.5, "deg": 90},
        "visibility": 8000,
        "clouds": {"all": 75}
    },
    "sydney": {
        "name": "Sydney",
        "sys": {"country": "AU"},
        "main": {
            "temp": 19.3,
            "feels_like": 18.7,
            "humidity": 65,
            "pressure": 1020,
            "temp_min": 16.0,
            "temp_max": 22.0
        },
        "weather": [{"description": "partly cloudy", "main": "Clouds"}],
        "wind": {"speed": 5.2, "deg": 135},
        "visibility": 12000,
        "clouds": {"all": 40}
    },
    "paris": {
        "name": "Paris",
        "sys": {"country": "FR"},
        "main": {
            "temp": 18.0,
            "feels_like": 17.2,
            "humidity": 68,
            "pressure": 1015,
            "temp_min": 14.5,
            "temp_max": 21.0
        },
        "weather": [{"description": "few clouds", "main": "Clouds"}],
        "wind": {"speed": 3.0, "deg": 200},
        "visibility": 10000,
        "clouds": {"all": 20}
    }
}


def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32


def get_wind_direction(degrees):
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]
    index = round(degrees / 22.5) % 16
    return directions[index]


def fetch_weather_live(city, api_key):
    params = urllib.parse.urlencode({
        "q": city,
        "appid": api_key,
        "units": "metric"
    })
    url = f"{BASE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read()
            return json.loads(data), None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "Invalid API key. Please check your API key and try again."
        elif e.code == 404:
            return None, f"City '{city}' not found. Please check the city name."
        elif e.code == 429:
            return None, "Too many requests. You have exceeded your API call limit."
        else:
            return None, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"Network error: {e.reason}"
    except json.JSONDecodeError:
        return None, "Failed to parse weather data from server."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def fetch_weather_demo(city):
    city_lower = city.strip().lower()
    if city_lower in DEMO_DATA:
        return DEMO_DATA[city_lower], None
    else:
        available = ", ".join(c.title() for c in DEMO_DATA.keys())
        return None, (
            f"City '{city}' not found in demo data.\n"
            f"Available demo cities: {available}\n"
            "To use a real API, set your API key at the top of app.py."
        )


def display_weather(data):
    city_name = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "Unknown")
    main = data.get("main", {})
    weather_list = data.get("weather", [{}])
    weather_desc = weather_list[0].get("description", "Unknown").title()
    weather_main = weather_list[0].get("main", "Unknown")
    wind = data.get("wind", {})
    visibility = data.get("visibility", None)
    clouds = data.get("clouds", {}).get("all", None)

    temp_c = main.get("temp", 0)
    feels_like_c = main.get("feels_like", 0)
    temp_min_c = main.get("temp_min", 0)
    temp_max_c = main.get("temp_max", 0)
    humidity = main.get("humidity", 0)
    pressure = main.get("pressure", 0)

    temp_f = celsius_to_fahrenheit(temp_c)
    feels_like_f = celsius_to_fahrenheit(feels_like_c)
    temp_min_f = celsius_to_fahrenheit(temp_min_c)
    temp_max_f = celsius_to_fahrenheit(temp_max_c)

    wind_speed = wind.get("speed", 0)
    wind_deg = wind.get("deg", 0)
    wind_dir = get_wind_direction(wind_deg)

    separator = "=" * 50
    thin_sep = "-" * 50

    print()
    print(separator)
    print(f"  Weather for {city_name}, {country}")
    print(separator)
    print(f"  Condition     : {weather_desc} ({weather_main})")
    print(thin_sep)
    print(f"  Temperature   : {temp_c:.1f}°C / {temp_f:.1f}°F")
    print(f"  Feels Like    : {feels_like_c:.1f}°C / {feels_like_f:.1f}°F")
    print(f"  Min / Max     : {temp_min_c:.1f}°C / {temp_min_f:.1f}°F  —  {temp_max_c:.1f}°C / {temp_max_f:.1f}°F")
    print(thin_sep)
    print(f"  Humidity      : {humidity}%")
    print(f"  Pressure      : {pressure} hPa")
    if visibility is not None:
        vis_km = visibility / 1000
        print(f"  Visibility    : {vis_km:.1f} km")
    if clouds is not None:
        print(f"  Cloud Cover   : {clouds}%")
    print(f"  Wind          : {wind_speed} m/s from {wind_dir} ({wind_deg}°)")
    print(separator)
    print()


def print_banner():
    print()
    print("=" * 50)
    print("       Simple Weather App")
    print("=" * 50)
    print()


def print_instructions():
    is_demo = API_KEY == "demo"
    if is_demo:
        print("  Mode: DEMO (using built-in sample data)")
        print()
        print("  Demo cities available:")
        for city in DEMO_DATA.keys():
            print(f"    - {city.title()}")
        print()
        print("  To use real weather data:")
        print("  1. Get a free API key from https://openweathermap.org/api")
        print("  2. Set API_KEY at the top of app.py")
        print()
    else:
        print("  Mode: LIVE (using OpenWeatherMap API)")
        print()


def run_app():
    print_banner()
    print_instructions()

    while True:
        try:
            city = input("  Enter city name (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not city:
            print("  Please enter a city name.\n")
            continue

        if city.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        print(f"\n  Fetching weather for '{city}'...")

        if API_KEY == "demo":
            data, error = fetch_weather_demo(city)
        else:
            data, error = fetch_weather_live(city, API_KEY)

        if error:
            print(f"\n  Error: {error}\n")
        else:
            display_weather(data)

        print("  Search another city or type 'quit' to exit.")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        city_arg = " ".join(sys.argv[1:])
        print_banner()
        if API_KEY == "demo":
            data, error = fetch_weather_demo(city_arg)
        else:
            data, error = fetch_weather_live(city_arg, API_KEY)
        if error:
            print(f"  Error: {error}")
            sys.exit(1)
        else:
            display_weather(data)
    else:
        run_app()
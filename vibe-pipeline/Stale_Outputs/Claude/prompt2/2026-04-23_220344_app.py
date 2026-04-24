import requests
import sys

API_KEY = "demo"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

DEMO_DATA = {
    "london": {
        "city": "London",
        "country": "GB",
        "temperature": 15.2,
        "feels_like": 13.8,
        "humidity": 76,
        "pressure": 1012,
        "description": "overcast clouds",
        "wind_speed": 5.1,
        "wind_direction": 230,
        "visibility": 10000,
        "min_temp": 12.0,
        "max_temp": 18.0,
    },
    "new york": {
        "city": "New York",
        "country": "US",
        "temperature": 22.5,
        "feels_like": 21.0,
        "humidity": 60,
        "pressure": 1015,
        "description": "clear sky",
        "wind_speed": 3.6,
        "wind_direction": 180,
        "visibility": 10000,
        "min_temp": 18.0,
        "max_temp": 26.0,
    },
    "tokyo": {
        "city": "Tokyo",
        "country": "JP",
        "temperature": 28.0,
        "feels_like": 30.5,
        "humidity": 80,
        "pressure": 1008,
        "description": "light rain",
        "wind_speed": 4.2,
        "wind_direction": 90,
        "visibility": 8000,
        "min_temp": 24.0,
        "max_temp": 31.0,
    },
    "sydney": {
        "city": "Sydney",
        "country": "AU",
        "temperature": 19.0,
        "feels_like": 18.2,
        "humidity": 65,
        "pressure": 1020,
        "description": "partly cloudy",
        "wind_speed": 6.0,
        "wind_direction": 270,
        "visibility": 10000,
        "min_temp": 15.0,
        "max_temp": 22.0,
    },
    "paris": {
        "city": "Paris",
        "country": "FR",
        "temperature": 17.5,
        "feels_like": 16.0,
        "humidity": 70,
        "pressure": 1010,
        "description": "scattered clouds",
        "wind_speed": 4.5,
        "wind_direction": 200,
        "visibility": 9000,
        "min_temp": 14.0,
        "max_temp": 20.0,
    },
}


def get_wind_direction_label(degrees):
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]
    index = round(degrees / 22.5) % 16
    return directions[index]


def fetch_weather_live(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 401:
            return None, "Invalid API key. Using demo mode instead."
        if response.status_code == 404:
            return None, f"City '{city}' not found."
        if response.status_code != 200:
            return None, f"API error: HTTP {response.status_code}"
        data = response.json()
        weather = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"].get("deg", 0),
            "visibility": data.get("visibility", 0),
            "min_temp": data["main"]["temp_min"],
            "max_temp": data["main"]["temp_max"],
        }
        return weather, None
    except requests.exceptions.ConnectionError:
        return None, "No internet connection."
    except requests.exceptions.Timeout:
        return None, "Request timed out."
    except Exception as e:
        return None, f"Unexpected error: {e}"


def fetch_weather_demo(city):
    key = city.strip().lower()
    if key in DEMO_DATA:
        return DEMO_DATA[key], None
    return None, f"City '{city}' not found in demo data. Try: London, New York, Tokyo, Sydney, Paris."


def display_weather(weather):
    separator = "=" * 50
    print()
    print(separator)
    print(f"  Weather for {weather['city']}, {weather['country']}")
    print(separator)
    print(f"  Description   : {weather['description'].capitalize()}")
    print(f"  Temperature   : {weather['temperature']:.1f}°C  (feels like {weather['feels_like']:.1f}°C)")
    print(f"  Min / Max     : {weather['min_temp']:.1f}°C / {weather['max_temp']:.1f}°C")
    print(f"  Humidity      : {weather['humidity']}%")
    print(f"  Pressure      : {weather['pressure']} hPa")
    wind_label = get_wind_direction_label(weather['wind_direction'])
    print(f"  Wind          : {weather['wind_speed']:.1f} m/s  {wind_label} ({weather['wind_direction']}°)")
    visibility_km = weather['visibility'] / 1000 if weather['visibility'] else 0
    print(f"  Visibility    : {visibility_km:.1f} km")
    print(separator)
    print()


def run_app(use_demo):
    print()
    print("  ╔══════════════════════════════╗")
    print("  ║     🌤  Weather App          ║")
    print("  ╚══════════════════════════════╝")
    if use_demo:
        print()
        print("  [DEMO MODE] Using built-in sample data.")
        print("  Available cities: London, New York, Tokyo, Sydney, Paris")
        print()
        print("  To use live data, replace API_KEY at the top of app.py")
        print("  with a real key from https://openweathermap.org/api")

    while True:
        print()
        city = input("  Enter city name (or 'quit' to exit): ").strip()
        if city.lower() in ("quit", "exit", "q"):
            print()
            print("  Goodbye!")
            print()
            break
        if not city:
            print("  Please enter a city name.")
            continue

        if use_demo:
            weather, error = fetch_weather_demo(city)
        else:
            weather, error = fetch_weather_live(city)

        if error:
            print(f"\n  Error: {error}")
            if not use_demo and "Invalid API key" in error:
                print("  Falling back to demo mode for this request.")
                weather, error2 = fetch_weather_demo(city)
                if error2:
                    print(f"  Demo error: {error2}")
                    continue
                display_weather(weather)
        else:
            display_weather(weather)


def main():
    use_demo = (API_KEY == "demo")
    try:
        run_app(use_demo)
    except KeyboardInterrupt:
        print()
        print()
        print("  Interrupted. Goodbye!")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
import urllib.request
import urllib.parse
import json
import sys

API_KEY = "demo"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    params = urllib.parse.urlencode({
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    })
    url = f"{BASE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            error_data = json.loads(body)
            print(f"\nAPI Error {e.code}: {error_data.get('message', 'Unknown error')}")
        except Exception:
            print(f"\nHTTP Error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"\nNetwork Error: {e.reason}")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None

def display_weather(data, city):
    if not data:
        return
    if data.get("cod") != 200:
        print(f"Error: {data.get('message', 'Could not retrieve weather data')}")
        return

    name = data.get("name", city)
    country = data.get("sys", {}).get("country", "")
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "N/A").capitalize()
    main = data.get("main", {})
    temp = main.get("temp", "N/A")
    feels_like = main.get("feels_like", "N/A")
    temp_min = main.get("temp_min", "N/A")
    temp_max = main.get("temp_max", "N/A")
    humidity = main.get("humidity", "N/A")
    pressure = main.get("pressure", "N/A")
    wind = data.get("wind", {})
    wind_speed = wind.get("speed", "N/A")
    wind_deg = wind.get("deg", "N/A")
    visibility = data.get("visibility", "N/A")
    if visibility != "N/A":
        visibility = f"{int(visibility) / 1000:.1f} km"
    clouds = data.get("clouds", {}).get("all", "N/A")

    print("\n" + "=" * 50)
    print(f"  Weather for {name}, {country}")
    print("=" * 50)
    print(f"  Condition    : {description}")
    print(f"  Temperature  : {temp}°C (feels like {feels_like}°C)")
    print(f"  Min / Max    : {temp_min}°C / {temp_max}°C")
    print(f"  Humidity     : {humidity}%")
    print(f"  Pressure     : {pressure} hPa")
    print(f"  Wind Speed   : {wind_speed} m/s at {wind_deg}°")
    print(f"  Visibility   : {visibility}")
    print(f"  Cloud Cover  : {clouds}%")
    print("=" * 50)

def check_api_key():
    if API_KEY == "demo" or not API_KEY:
        print("\n" + "!" * 60)
        print("  WARNING: No valid API key set.")
        print("  To use this app:")
        print("  1. Sign up at https://openweathermap.org/api")
        print("  2. Get your free API key")
        print("  3. Replace the API_KEY value at the top of app.py")
        print("!" * 60)
        return False
    return True

def main():
    print("=" * 50)
    print("        Simple Weather App")
    print("  Powered by OpenWeatherMap API")
    print("=" * 50)

    api_ok = check_api_key()

    if not api_ok:
        demo_data = {
            "cod": 200,
            "name": "London",
            "sys": {"country": "GB"},
            "weather": [{"description": "light rain"}],
            "main": {
                "temp": 12.5,
                "feels_like": 10.2,
                "temp_min": 10.0,
                "temp_max": 14.0,
                "humidity": 82,
                "pressure": 1012
            },
            "wind": {"speed": 4.6, "deg": 220},
            "visibility": 8000,
            "clouds": {"all": 75}
        }
        print("\n  Showing demo data for London (API key not set):")
        display_weather(demo_data, "London")
        print("\n  Set a real API key in app.py to fetch live data.")
        return

    while True:
        print()
        city = input("Enter city name (or 'quit' to exit): ").strip()
        if city.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not city:
            print("Please enter a valid city name.")
            continue
        print(f"\nFetching weather for '{city}'...")
        data = get_weather(city)
        display_weather(data, city)

if __name__ == "__main__":
    main()
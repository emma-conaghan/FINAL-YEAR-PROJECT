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
            data = response.read()
            return json.loads(data)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("\n[ERROR] Invalid API key.")
            print("To use this app, please:")
            print("  1. Go to https://openweathermap.org/api and create a free account")
            print("  2. Get your API key from the dashboard")
            print("  3. Replace the API_KEY variable at the top of app.py with your key")
            return None
        elif e.code == 404:
            print(f"\n[ERROR] City '{city}' not found. Please check the spelling and try again.")
            return None
        else:
            print(f"\n[ERROR] HTTP error occurred: {e.code} - {e.reason}")
            return None
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Network error: {e.reason}")
        print("Please check your internet connection and try again.")
        return None
    except json.JSONDecodeError:
        print("\n[ERROR] Failed to parse response from weather API.")
        return None
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return None

def display_weather(data, city):
    if not data:
        return
    print("\n" + "=" * 50)
    print(f"  Weather Report for: {data.get('name', city)}, {data.get('sys', {}).get('country', '')}")
    print("=" * 50)

    weather_list = data.get("weather", [{}])
    weather_desc = weather_list[0].get("description", "N/A").capitalize() if weather_list else "N/A"

    main = data.get("main", {})
    wind = data.get("wind", {})
    clouds = data.get("clouds", {})
    visibility = data.get("visibility", None)
    sys_info = data.get("sys", {})

    temp = main.get("temp", "N/A")
    feels_like = main.get("feels_like", "N/A")
    temp_min = main.get("temp_min", "N/A")
    temp_max = main.get("temp_max", "N/A")
    humidity = main.get("humidity", "N/A")
    pressure = main.get("pressure", "N/A")
    wind_speed = wind.get("speed", "N/A")
    wind_deg = wind.get("deg", "N/A")
    cloud_cover = clouds.get("all", "N/A")

    print(f"  Condition     : {weather_desc}")
    print(f"  Temperature   : {temp}°C")
    print(f"  Feels Like    : {feels_like}°C")
    print(f"  Min / Max     : {temp_min}°C / {temp_max}°C")
    print(f"  Humidity      : {humidity}%")
    print(f"  Pressure      : {pressure} hPa")
    print(f"  Wind Speed    : {wind_speed} m/s")
    print(f"  Wind Direction: {wind_deg}°")
    print(f"  Cloud Cover   : {cloud_cover}%")
    if visibility is not None:
        print(f"  Visibility    : {visibility / 1000:.1f} km")
    print("=" * 50 + "\n")

def main():
    print("=" * 50)
    print("       Simple Weather App (OpenWeatherMap)")
    print("=" * 50)

    if API_KEY == "demo":
        print("\n[NOTICE] You are using a placeholder API key.")
        print("The app will attempt to connect but will likely return a 401 error.")
        print("Please replace API_KEY in app.py with your real OpenWeatherMap key.\n")

    while True:
        try:
            city = input("Enter city name (or 'quit' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)

        if city.lower() in ("quit", "exit", "q"):
            print("Exiting. Goodbye!")
            sys.exit(0)

        if not city:
            print("[WARNING] Please enter a valid city name.\n")
            continue

        print(f"\nFetching weather for '{city}'...")
        weather_data = get_weather(city)
        display_weather(weather_data, city)

        try:
            again = input("Check another city? (yes/no): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)

        if again not in ("yes", "y"):
            print("Exiting. Goodbye!")
            break

if __name__ == "__main__":
    main()
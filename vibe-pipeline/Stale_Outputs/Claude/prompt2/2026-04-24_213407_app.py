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
            print(f"API Error {e.code}: {error_data.get('message', 'Unknown error')}")
        except Exception:
            print(f"HTTP Error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def display_weather(data):
    if not data:
        return
    city_name = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "Unknown")
    temp = data.get("main", {}).get("temp", "N/A")
    feels_like = data.get("main", {}).get("feels_like", "N/A")
    temp_min = data.get("main", {}).get("temp_min", "N/A")
    temp_max = data.get("main", {}).get("temp_max", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    pressure = data.get("main", {}).get("pressure", "N/A")
    description = ""
    weather_list = data.get("weather", [])
    if weather_list:
        description = weather_list[0].get("description", "N/A").capitalize()
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    wind_deg = data.get("wind", {}).get("deg", "N/A")
    visibility = data.get("visibility", "N/A")
    if isinstance(visibility, (int, float)):
        visibility = f"{visibility / 1000:.1f} km"
    cloudiness = data.get("clouds", {}).get("all", "N/A")

    print("\n" + "=" * 50)
    print(f"  Weather for {city_name}, {country}")
    print("=" * 50)
    print(f"  Condition    : {description}")
    print(f"  Temperature  : {temp}°C")
    print(f"  Feels Like   : {feels_like}°C")
    print(f"  Min / Max    : {temp_min}°C / {temp_max}°C")
    print(f"  Humidity     : {humidity}%")
    print(f"  Pressure     : {pressure} hPa")
    print(f"  Wind Speed   : {wind_speed} m/s at {wind_deg}°")
    print(f"  Cloudiness   : {cloudiness}%")
    print(f"  Visibility   : {visibility}")
    print("=" * 50 + "\n")

def main():
    print("=" * 50)
    print("       Simple Weather App")
    print("=" * 50)
    print()

    if API_KEY == "demo":
        print("NOTE: You are using the demo API key placeholder.")
        print("To get real weather data, please:")
        print("  1. Sign up at https://openweathermap.org/api")
        print("  2. Get your free API key")
        print("  3. Replace the API_KEY value in this script")
        print()

    if len(sys.argv) > 1:
        city = " ".join(sys.argv[1:])
        print(f"Fetching weather for: {city}")
        data = get_weather(city)
        display_weather(data)
    else:
        while True:
            try:
                city = input("Enter city name (or 'quit' to exit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not city:
                print("Please enter a city name.")
                continue

            if city.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print(f"Fetching weather for: {city} ...")
            data = get_weather(city)
            display_weather(data)

if __name__ == "__main__":
    main()
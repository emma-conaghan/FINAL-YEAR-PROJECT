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
            err_data = json.loads(body)
            print(f"API Error {e.code}: {err_data.get('message', 'Unknown error')}")
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
    humidity = data.get("main", {}).get("humidity", "N/A")
    pressure = data.get("main", {}).get("pressure", "N/A")
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "N/A").capitalize() if weather_list else "N/A"
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    visibility = data.get("visibility", "N/A")
    if visibility != "N/A":
        visibility = f"{int(visibility) / 1000:.1f} km"

    print("\n" + "=" * 45)
    print(f"  Weather Report for {city_name}, {country}")
    print("=" * 45)
    print(f"  Condition   : {description}")
    print(f"  Temperature : {temp} °C")
    print(f"  Feels Like  : {feels_like} °C")
    print(f"  Humidity    : {humidity}%")
    print(f"  Pressure    : {pressure} hPa")
    print(f"  Wind Speed  : {wind_speed} m/s")
    print(f"  Visibility  : {visibility}")
    print("=" * 45 + "\n")

def prompt_for_api_key():
    print("\nNote: This app uses the OpenWeatherMap API.")
    print("You need a free API key from https://openweathermap.org/api")
    print("Sign up for free and use the 'Current Weather Data' API.\n")
    key = input("Enter your OpenWeatherMap API key (or press Enter to use 'demo'): ").strip()
    return key if key else "demo"

def main():
    global API_KEY

    print("=" * 45)
    print("       Simple Weather App")
    print("=" * 45)

    if API_KEY == "demo":
        API_KEY = prompt_for_api_key()

    while True:
        print("\nOptions:")
        print("  1. Get weather for a city")
        print("  2. Change API key")
        print("  3. Exit")
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            city = input("Enter city name: ").strip()
            if not city:
                print("City name cannot be empty.")
                continue
            print(f"\nFetching weather data for '{city}'...")
            data = get_weather(city)
            display_weather(data)
            if not data:
                print("Could not retrieve weather data. Please check the city name and API key.")

        elif choice == "2":
            new_key = input("Enter new API key: ").strip()
            if new_key:
                API_KEY = new_key
                print("API key updated.")
            else:
                print("No key entered, keeping existing key.")

        elif choice == "3":
            print("Goodbye!")
            sys.exit(0)

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
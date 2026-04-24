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
        error_body = e.read().decode("utf-8")
        try:
            error_json = json.loads(error_body)
            print(f"API Error {e.code}: {error_json.get('message', 'Unknown error')}")
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
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    wind_deg = data.get("wind", {}).get("deg", "N/A")
    visibility = data.get("visibility", "N/A")
    weather_desc = "N/A"
    weather_main = "N/A"
    if data.get("weather") and len(data["weather"]) > 0:
        weather_desc = data["weather"][0].get("description", "N/A").capitalize()
        weather_main = data["weather"][0].get("main", "N/A")
    clouds = data.get("clouds", {}).get("all", "N/A")

    print("\n" + "=" * 50)
    print(f"  Weather for {city_name}, {country}")
    print("=" * 50)
    print(f"  Condition   : {weather_main} - {weather_desc}")
    print(f"  Temperature : {temp}°C (Feels like {feels_like}°C)")
    print(f"  Humidity    : {humidity}%")
    print(f"  Pressure    : {pressure} hPa")
    print(f"  Wind Speed  : {wind_speed} m/s at {wind_deg}°")
    if visibility != "N/A":
        print(f"  Visibility  : {int(visibility) // 1000} km")
    else:
        print(f"  Visibility  : N/A")
    print(f"  Cloud Cover : {clouds}%")
    print("=" * 50 + "\n")

def setup_api_key():
    global API_KEY
    print("\nWeather App - Powered by OpenWeatherMap")
    print("-" * 40)
    print("NOTE: This app requires a free API key from https://openweathermap.org/api")
    print("Sign up for free at: https://home.openweathermap.org/users/sign_up")
    print("-" * 40)
    user_key = input("Enter your OpenWeatherMap API key (or press Enter to try with demo): ").strip()
    if user_key:
        API_KEY = user_key
        print(f"Using provided API key.")
    else:
        print("Using demo key - this may not work. Please get a free API key.")

def main():
    setup_api_key()
    print("\nType a city name to get weather info. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            city = input("Enter city name: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)
        if not city:
            print("Please enter a valid city name.")
            continue
        if city.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        print(f"\nFetching weather data for '{city}'...")
        data = get_weather(city)
        if data and data.get("cod") == 200:
            display_weather(data)
        elif data:
            code = data.get("cod")
            message = data.get("message", "Unknown error")
            print(f"Error {code}: {message}")
        else:
            print("Could not retrieve weather data. Please check the city name and try again.")

if __name__ == "__main__":
    main()
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
        error_body = e.read().decode("utf-8")
        try:
            error_data = json.loads(error_body)
            print(f"API Error: {error_data.get('message', 'Unknown error')}")
        except Exception:
            print(f"HTTP Error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def display_weather(data, city):
    if not data:
        return
    if data.get("cod") != 200:
        print(f"Error: {data.get('message', 'Could not retrieve weather data')}")
        return

    name = data.get("name", city)
    country = data.get("sys", {}).get("country", "")
    temp = data.get("main", {}).get("temp", "N/A")
    feels_like = data.get("main", {}).get("feels_like", "N/A")
    temp_min = data.get("main", {}).get("temp_min", "N/A")
    temp_max = data.get("main", {}).get("temp_max", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    pressure = data.get("main", {}).get("pressure", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    wind_deg = data.get("wind", {}).get("deg", "N/A")
    visibility = data.get("visibility", "N/A")
    description = ""
    weather_list = data.get("weather", [])
    if weather_list:
        description = weather_list[0].get("description", "").capitalize()
    cloudiness = data.get("clouds", {}).get("all", "N/A")

    print("\n" + "=" * 50)
    print(f"  Weather in {name}, {country}")
    print("=" * 50)
    print(f"  Condition    : {description}")
    print(f"  Temperature  : {temp}°C (feels like {feels_like}°C)")
    print(f"  Min / Max    : {temp_min}°C / {temp_max}°C")
    print(f"  Humidity     : {humidity}%")
    print(f"  Pressure     : {pressure} hPa")
    print(f"  Wind Speed   : {wind_speed} m/s at {wind_deg}°")
    print(f"  Cloudiness   : {cloudiness}%")
    if visibility != "N/A":
        print(f"  Visibility   : {int(visibility) // 1000} km")
    print("=" * 50)

def main():
    print("=" * 50)
    print("        Simple Weather App")
    print("=" * 50)
    print()
    print("NOTE: This app uses the OpenWeatherMap API.")
    print("      The default API key is 'demo' and may not work.")
    print("      To use a real key, sign up at https://openweathermap.org/api")
    print("      and replace the API_KEY variable at the top of this file.")
    print()

    if len(sys.argv) > 1:
        city = " ".join(sys.argv[1:])
    else:
        city = input("Enter city name: ").strip()

    if not city:
        print("No city entered. Exiting.")
        sys.exit(1)

    print(f"\nFetching weather for '{city}'...")
    data = get_weather(city)
    display_weather(data, city)

    print()
    while True:
        again = input("Search another city? (yes/no): ").strip().lower()
        if again in ("yes", "y"):
            city = input("Enter city name: ").strip()
            if not city:
                print("No city entered.")
                continue
            print(f"\nFetching weather for '{city}'...")
            data = get_weather(city)
            display_weather(data, city)
            print()
        elif again in ("no", "n", ""):
            print("Goodbye!")
            break
        else:
            print("Please enter yes or no.")

if __name__ == "__main__":
    main()
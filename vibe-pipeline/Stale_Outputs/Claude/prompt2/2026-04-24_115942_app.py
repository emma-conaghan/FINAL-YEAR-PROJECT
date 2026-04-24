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
            print(f"API Error {e.code}: {error_data.get('message', 'Unknown error')}")
        except Exception:
            print(f"HTTP Error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def display_weather(data):
    if not data:
        print("No weather data to display.")
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
    clouds = data.get("clouds", {}).get("all", "N/A")

    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "N/A") if weather_list else "N/A"
    main_weather = weather_list[0].get("main", "N/A") if weather_list else "N/A"

    sunrise_ts = data.get("sys", {}).get("sunrise")
    sunset_ts = data.get("sys", {}).get("sunset")

    import datetime
    if sunrise_ts:
        sunrise = datetime.datetime.utcfromtimestamp(sunrise_ts).strftime("%H:%M UTC")
    else:
        sunrise = "N/A"

    if sunset_ts:
        sunset = datetime.datetime.utcfromtimestamp(sunset_ts).strftime("%H:%M UTC")
    else:
        sunset = "N/A"

    print("\n" + "=" * 50)
    print(f"  Weather for {city_name}, {country}")
    print("=" * 50)
    print(f"  Condition    : {main_weather} ({description})")
    print(f"  Temperature  : {temp}°C (feels like {feels_like}°C)")
    print(f"  Humidity     : {humidity}%")
    print(f"  Pressure     : {pressure} hPa")
    print(f"  Wind Speed   : {wind_speed} m/s at {wind_deg}°")
    print(f"  Cloud Cover  : {clouds}%")
    if visibility != "N/A":
        print(f"  Visibility   : {int(visibility) / 1000:.1f} km")
    else:
        print(f"  Visibility   : N/A")
    print(f"  Sunrise      : {sunrise}")
    print(f"  Sunset       : {sunset}")
    print("=" * 50 + "\n")

def main():
    print("=" * 50)
    print("       Simple Weather App")
    print("=" * 50)
    print("NOTE: This app uses the OpenWeatherMap API.")
    print("      The default API key is a placeholder.")
    print("      Get a free key at https://openweathermap.org/api")
    print("      then replace API_KEY at the top of this file.")
    print("=" * 50 + "\n")

    if len(sys.argv) > 1:
        city = " ".join(sys.argv[1:])
        print(f"Looking up weather for: {city}")
        data = get_weather(city)
        display_weather(data)
    else:
        while True:
            try:
                city = input("Enter city name (or 'quit' to exit): ").strip()
                if city.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not city:
                    print("Please enter a city name.")
                    continue
                print(f"Fetching weather for '{city}'...")
                data = get_weather(city)
                display_weather(data)
                another = input("Check another city? (yes/no): ").strip().lower()
                if another not in ("yes", "y"):
                    print("Goodbye!")
                    break
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
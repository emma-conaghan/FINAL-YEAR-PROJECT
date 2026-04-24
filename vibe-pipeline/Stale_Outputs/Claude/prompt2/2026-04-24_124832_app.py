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
        if e.code == 401:
            print("\nERROR: Invalid API key.")
            print("Please sign up at https://openweathermap.org/api to get a free API key.")
            print("Then replace the API_KEY variable at the top of app.py with your key.")
        elif e.code == 404:
            print(f"\nERROR: City '{city}' not found. Please check the spelling and try again.")
        else:
            print(f"\nERROR: HTTP error occurred: {e.code} {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"\nERROR: Could not connect to the weather service. Check your internet connection.")
        print(f"Details: {e.reason}")
        return None
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        return None

def display_weather(data, city):
    if not data:
        return

    if data.get("cod") != 200:
        message = data.get("message", "Unknown error")
        print(f"\nERROR: {message}")
        return

    print("\n" + "=" * 50)
    print(f"  Weather for: {data.get('name', city)}, {data.get('sys', {}).get('country', '')}")
    print("=" * 50)

    weather_list = data.get("weather", [{}])
    weather_desc = weather_list[0].get("description", "N/A").capitalize() if weather_list else "N/A"
    weather_main = weather_list[0].get("main", "N/A") if weather_list else "N/A"

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

    print(f"  Condition     : {weather_main} - {weather_desc}")
    print(f"  Temperature   : {temp}°C (Feels like {feels_like}°C)")
    print(f"  Min / Max     : {temp_min}°C / {temp_max}°C")
    print(f"  Humidity      : {humidity}%")
    print(f"  Pressure      : {pressure} hPa")
    print(f"  Wind Speed    : {wind_speed} m/s at {wind_deg}°")
    print(f"  Cloud Cover   : {cloud_cover}%")

    if visibility is not None:
        vis_km = round(visibility / 1000, 1)
        print(f"  Visibility    : {vis_km} km")

    sunrise = sys_info.get("sunrise")
    sunset = sys_info.get("sunset")
    if sunrise and sunset:
        import datetime
        sunrise_time = datetime.datetime.fromtimestamp(sunrise).strftime("%H:%M:%S")
        sunset_time = datetime.datetime.fromtimestamp(sunset).strftime("%H:%M:%S")
        print(f"  Sunrise       : {sunrise_time}")
        print(f"  Sunset        : {sunset_time}")

    print("=" * 50)

def main():
    print("=" * 50)
    print("       Simple Weather Application")
    print("=" * 50)
    print("\nNOTE: This app uses the OpenWeatherMap API.")
    print("      A valid API key is required.")
    print("      Get a free key at: https://openweathermap.org/api")
    print("      Then set API_KEY in app.py\n")

    if API_KEY == "demo":
        print("WARNING: You are using the placeholder API key 'demo'.")
        print("         Requests will likely fail with a 401 error.")
        print("         Please update API_KEY in app.py with your real key.\n")

    while True:
        try:
            city = input("Enter city name (or 'quit' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            sys.exit(0)

        if not city:
            print("Please enter a valid city name.")
            continue

        if city.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            sys.exit(0)

        print(f"\nFetching weather data for '{city}'...")
        data = get_weather(city)
        display_weather(data, city)
        print()

if __name__ == "__main__":
    main()
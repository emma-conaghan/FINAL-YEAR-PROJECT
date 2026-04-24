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
            print("\nERROR: Invalid API key.")
            print("Please get a free API key from https://openweathermap.org/api")
            print("Then replace the API_KEY variable at the top of app.py with your key.")
        elif e.code == 404:
            print(f"\nERROR: City '{city}' not found. Please check the spelling and try again.")
        else:
            print(f"\nERROR: HTTP error occurred: {e.code} {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"\nERROR: Failed to reach the server. Check your internet connection.\nDetails: {e.reason}")
        return None
    except json.JSONDecodeError:
        print("\nERROR: Failed to parse the response from the weather API.")
        return None

def display_weather(data, city):
    if not data:
        return

    name = data.get("name", city)
    country = data.get("sys", {}).get("country", "N/A")
    temp = data.get("main", {}).get("temp", "N/A")
    feels_like = data.get("main", {}).get("feels_like", "N/A")
    temp_min = data.get("main", {}).get("temp_min", "N/A")
    temp_max = data.get("main", {}).get("temp_max", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    pressure = data.get("main", {}).get("pressure", "N/A")
    visibility = data.get("visibility", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    wind_deg = data.get("wind", {}).get("deg", "N/A")
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "N/A").capitalize() if weather_list else "N/A"
    clouds = data.get("clouds", {}).get("all", "N/A")

    if visibility != "N/A":
        visibility_km = round(visibility / 1000, 1)
    else:
        visibility_km = "N/A"

    print("\n" + "=" * 50)
    print(f"  Weather Report: {name}, {country}")
    print("=" * 50)
    print(f"  Condition      : {description}")
    print(f"  Temperature    : {temp} °C")
    print(f"  Feels Like     : {feels_like} °C")
    print(f"  Min / Max      : {temp_min} °C / {temp_max} °C")
    print(f"  Humidity       : {humidity} %")
    print(f"  Pressure       : {pressure} hPa")
    print(f"  Visibility     : {visibility_km} km")
    print(f"  Wind Speed     : {wind_speed} m/s")
    print(f"  Wind Direction : {wind_deg}°")
    print(f"  Cloud Cover    : {clouds} %")
    print("=" * 50)

def main():
    print("=" * 50)
    print("       Simple Weather App")
    print("=" * 50)

    if API_KEY == "demo":
        print("\nNOTICE: You are using a placeholder API key.")
        print("To use this app, please:")
        print("  1. Visit https://openweathermap.org/api")
        print("  2. Sign up for a free account")
        print("  3. Copy your API key")
        print("  4. Replace the API_KEY variable at the top of app.py\n")

    if len(sys.argv) > 1:
        city = " ".join(sys.argv[1:])
        print(f"Fetching weather for: {city}")
        data = get_weather(city)
        display_weather(data, city)
    else:
        while True:
            try:
                city = input("\nEnter city name (or 'quit' to exit): ").strip()
            except (KeyboardInterrupt, EOFError):
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
            display_weather(data, city)

if __name__ == "__main__":
    main()
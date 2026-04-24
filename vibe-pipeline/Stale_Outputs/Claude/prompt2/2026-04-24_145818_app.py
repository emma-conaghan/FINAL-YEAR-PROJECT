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
        except json.JSONDecodeError:
            print(f"HTTP Error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"Connection Error: {e.reason}")
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
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "N/A").capitalize() if weather_list else "N/A"
    main_weather = weather_list[0].get("main", "N/A") if weather_list else "N/A"
    clouds = data.get("clouds", {}).get("all", "N/A")

    print("\n" + "=" * 50)
    print(f"  Weather in {city_name}, {country}")
    print("=" * 50)
    print(f"  Condition   : {description} ({main_weather})")
    print(f"  Temperature : {temp} °C")
    print(f"  Feels Like  : {feels_like} °C")
    print(f"  Humidity    : {humidity}%")
    print(f"  Pressure    : {pressure} hPa")
    print(f"  Wind Speed  : {wind_speed} m/s")
    print(f"  Wind Dir    : {wind_deg}°")
    print(f"  Cloud Cover : {clouds}%")
    if visibility != "N/A":
        print(f"  Visibility  : {int(visibility) // 1000} km")
    else:
        print(f"  Visibility  : N/A")
    print("=" * 50 + "\n")

def main():
    print("=" * 50)
    print("       Simple Weather App")
    print("=" * 50)
    print("NOTE: This app uses the OpenWeatherMap API.")
    print("      The default API key is 'demo' which may")
    print("      not work. Get a free key at:")
    print("      https://openweathermap.org/api")
    print("      Then replace API_KEY in this file.\n")

    if len(sys.argv) > 1:
        city = " ".join(sys.argv[1:])
        print(f"Looking up weather for: {city}")
        data = get_weather(city)
        display_weather(data)
    else:
        while True:
            try:
                city = input("Enter city name (or 'quit' to exit): ").strip()
                if not city:
                    print("Please enter a valid city name.")
                    continue
                if city.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                data = get_weather(city)
                if data:
                    display_weather(data)
                else:
                    print("Could not retrieve weather data. Please try again.\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
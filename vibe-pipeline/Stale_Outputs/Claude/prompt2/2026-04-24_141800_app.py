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
        print(f"Network Error: {e.reason}")
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
    weather_main = weather_list[0].get("main", "N/A") if weather_list else "N/A"
    clouds = data.get("clouds", {}).get("all", "N/A")

    print("\n" + "=" * 50)
    print(f"  Weather for {city_name}, {country}")
    print("=" * 50)
    print(f"  Condition    : {weather_main} - {description}")
    print(f"  Temperature  : {temp}°C")
    print(f"  Feels Like   : {feels_like}°C")
    print(f"  Humidity     : {humidity}%")
    print(f"  Pressure     : {pressure} hPa")
    print(f"  Wind Speed   : {wind_speed} m/s")
    print(f"  Wind Degree  : {wind_deg}°")
    if visibility != "N/A":
        print(f"  Visibility   : {int(visibility) // 1000} km")
    else:
        print(f"  Visibility   : N/A")
    print(f"  Cloud Cover  : {clouds}%")
    print("=" * 50 + "\n")

def get_demo_weather(city):
    demo_data = {
        "name": city,
        "sys": {"country": "US"},
        "main": {
            "temp": 22.5,
            "feels_like": 21.0,
            "humidity": 60,
            "pressure": 1013
        },
        "wind": {"speed": 5.5, "deg": 180},
        "visibility": 10000,
        "weather": [{"main": "Clear", "description": "clear sky"}],
        "clouds": {"all": 10}
    }
    return demo_data

def main():
    print("=" * 50)
    print("       Simple Weather Application")
    print("=" * 50)
    print()
    print("NOTE: This app uses the OpenWeatherMap API.")
    print("      To use a real API key, replace 'demo' in")
    print("      the API_KEY variable with your key from:")
    print("      https://openweathermap.org/api")
    print()
    print("      Running in DEMO mode with sample data.")
    print()

    use_demo = API_KEY == "demo"

    while True:
        try:
            city = input("Enter city name (or 'quit' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not city:
            print("Please enter a valid city name.\n")
            continue

        if city.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            sys.exit(0)

        if use_demo:
            print(f"\nFetching demo weather data for '{city}'...")
            data = get_demo_weather(city)
        else:
            print(f"\nFetching weather data for '{city}'...")
            data = get_weather(city)

        if data:
            display_weather(data)
        else:
            print("Could not retrieve weather data. Please try again.\n")

if __name__ == "__main__":
    main()
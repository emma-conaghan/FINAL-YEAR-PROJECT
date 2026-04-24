import requests
import sys

API_KEY = "demo"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

def kelvin_to_fahrenheit(kelvin):
    return (kelvin - 273.15) * 9/5 + 32

def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            print("\nError: Invalid API key.")
            print("To use this app with real data:")
            print("1. Sign up at https://openweathermap.org/api")
            print("2. Get your free API key")
            print("3. Replace the API_KEY value in app.py with your key")
            return None
        elif response.status_code == 404:
            print(f"\nError: City '{city}' not found. Please check the spelling and try again.")
            return None
        else:
            print(f"\nError: Received status code {response.status_code} from API.")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the weather service.")
        print("Please check your internet connection and try again.")
        return None
    except requests.exceptions.Timeout:
        print("\nError: The request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\nError: An unexpected error occurred: {e}")
        return None

def display_weather(data, city_input):
    city_name = data.get("name", city_input)
    country = data.get("sys", {}).get("country", "")
    temp_kelvin = data.get("main", {}).get("temp", 0)
    feels_like_kelvin = data.get("main", {}).get("feels_like", 0)
    temp_min_kelvin = data.get("main", {}).get("temp_min", 0)
    temp_max_kelvin = data.get("main", {}).get("temp_max", 0)
    humidity = data.get("main", {}).get("humidity", 0)
    pressure = data.get("main", {}).get("pressure", 0)
    wind_speed = data.get("wind", {}).get("speed", 0)
    wind_deg = data.get("wind", {}).get("deg", 0)
    visibility = data.get("visibility", 0)
    clouds = data.get("clouds", {}).get("all", 0)
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "N/A").capitalize() if weather_list else "N/A"
    weather_main = weather_list[0].get("main", "N/A") if weather_list else "N/A"

    temp_c = kelvin_to_celsius(temp_kelvin)
    temp_f = kelvin_to_fahrenheit(temp_kelvin)
    feels_like_c = kelvin_to_celsius(feels_like_kelvin)
    feels_like_f = kelvin_to_fahrenheit(feels_like_kelvin)
    temp_min_c = kelvin_to_celsius(temp_min_kelvin)
    temp_max_c = kelvin_to_celsius(temp_max_kelvin)
    temp_min_f = kelvin_to_fahrenheit(temp_min_kelvin)
    temp_max_f = kelvin_to_fahrenheit(temp_max_kelvin)

    wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    wind_direction_label = wind_directions[round(wind_deg / 45) % 8]

    print("\n" + "=" * 50)
    print(f"  Weather for {city_name}, {country}")
    print("=" * 50)
    print(f"  Condition     : {weather_main} - {description}")
    print(f"  Temperature   : {temp_c:.1f}°C / {temp_f:.1f}°F")
    print(f"  Feels Like    : {feels_like_c:.1f}°C / {feels_like_f:.1f}°F")
    print(f"  Min / Max     : {temp_min_c:.1f}°C / {temp_min_f:.1f}°F  —  {temp_max_c:.1f}°C / {temp_max_f:.1f}°F")
    print(f"  Humidity      : {humidity}%")
    print(f"  Pressure      : {pressure} hPa")
    print(f"  Wind          : {wind_speed} m/s ({wind_direction_label})")
    print(f"  Cloud Cover   : {clouds}%")
    if visibility:
        print(f"  Visibility    : {visibility / 1000:.1f} km")
    print("=" * 50)

def show_demo_weather(city):
    print(f"\n[DEMO MODE] Showing sample weather data for '{city}'")
    print("(To get real data, replace API_KEY in app.py with your OpenWeatherMap API key)")
    demo_data = {
        "name": city.title(),
        "sys": {"country": "XX"},
        "main": {
            "temp": 295.15,
            "feels_like": 294.15,
            "temp_min": 292.15,
            "temp_max": 298.15,
            "humidity": 65,
            "pressure": 1013,
        },
        "wind": {"speed": 5.5, "deg": 180},
        "visibility": 10000,
        "clouds": {"all": 20},
        "weather": [{"main": "Clear", "description": "clear sky"}],
    }
    display_weather(demo_data, city)

def main():
    print("=" * 50)
    print("        Simple Weather Application")
    print("=" * 50)

    if API_KEY == "demo":
        print("\nNOTE: Running in DEMO MODE.")
        print("To use real weather data:")
        print("  1. Sign up at https://openweathermap.org/api")
        print("  2. Get your free API key")
        print("  3. Open app.py and replace the value of API_KEY with your key")

    while True:
        print()
        city = input("Enter a city name (or 'quit' to exit): ").strip()

        if city.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if not city:
            print("Please enter a valid city name.")
            continue

        if API_KEY == "demo":
            show_demo_weather(city)
        else:
            data = get_weather(city)
            if data:
                display_weather(data, city)

        print()
        again = input("Would you like to check another city? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Error: The 'requests' library is not installed.")
        print("Please run:  pip install requests")
        sys.exit(1)
    main()
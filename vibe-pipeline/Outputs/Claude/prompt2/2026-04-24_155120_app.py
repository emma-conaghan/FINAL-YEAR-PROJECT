import requests
import sys

API_KEY = "demo"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            print("Error: Invalid API key. Please set a valid OpenWeatherMap API key.")
            print("Get a free key at https://openweathermap.org/api")
            return None
        elif response.status_code == 404:
            print(f"Error: City '{city}' not found. Please check the city name and try again.")
            return None
        else:
            print(f"Error: Received status code {response.status_code} from the weather API.")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the weather API. Check your internet connection.")
        return None
    except requests.exceptions.Timeout:
        print("Error: The request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: An unexpected error occurred: {e}")
        return None

def display_weather(data, city):
    if not data:
        return

    city_name = data.get("name", city)
    country = data.get("sys", {}).get("country", "N/A")
    temp = data.get("main", {}).get("temp", "N/A")
    feels_like = data.get("main", {}).get("feels_like", "N/A")
    temp_min = data.get("main", {}).get("temp_min", "N/A")
    temp_max = data.get("main", {}).get("temp_max", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    pressure = data.get("main", {}).get("pressure", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    wind_deg = data.get("wind", {}).get("deg", "N/A")
    visibility = data.get("visibility", "N/A")
    description = "N/A"
    if data.get("weather"):
        description = data["weather"][0].get("description", "N/A").capitalize()
    clouds = data.get("clouds", {}).get("all", "N/A")

    print("\n" + "=" * 50)
    print(f"  Weather for {city_name}, {country}")
    print("=" * 50)
    print(f"  Condition    : {description}")
    print(f"  Temperature  : {temp}°C (Feels like {feels_like}°C)")
    print(f"  Min / Max    : {temp_min}°C / {temp_max}°C")
    print(f"  Humidity     : {humidity}%")
    print(f"  Pressure     : {pressure} hPa")
    print(f"  Wind Speed   : {wind_speed} m/s at {wind_deg}°")
    if visibility != "N/A":
        print(f"  Visibility   : {int(visibility) // 1000} km")
    print(f"  Cloud Cover  : {clouds}%")
    print("=" * 50 + "\n")

def setup_api_key():
    global API_KEY
    print("\nNOTE: This app uses the OpenWeatherMap API.")
    print("The default API key is a placeholder and may not work.")
    print("Get a free API key at: https://openweathermap.org/api\n")
    user_key = input("Enter your OpenWeatherMap API key (or press Enter to use demo key): ").strip()
    if user_key:
        API_KEY = user_key
        print("API key set successfully.\n")
    else:
        print("Using demo key. Results may be limited or unavailable.\n")

def main():
    print("=" * 50)
    print("       Simple Weather App")
    print("=" * 50)

    setup_api_key()

    while True:
        city = input("Enter city name (or 'quit' to exit): ").strip()
        if city.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            sys.exit(0)
        if not city:
            print("Please enter a valid city name.")
            continue
        data = get_weather(city)
        display_weather(data, city)

        again = input("Search for another city? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
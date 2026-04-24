import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # Sample API key for openweathermap.org free tier (demo)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if not data:
        print("No data to display.")
        return
    if data.get("cod") != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    city = data.get("name")
    temp = data["main"]["temp"]
    weather = data["weather"][0]["description"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]

    print(f"Weather for {city}:")
    print(f"Temperature: {temp}°C")
    print(f"Condition: {weather}")
    print(f"Humidity: {humidity}%")
    print(f"Wind speed: {wind_speed} m/s")

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if not city:
        print("City name cannot be empty.")
        return
    weather_data = get_weather(city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()
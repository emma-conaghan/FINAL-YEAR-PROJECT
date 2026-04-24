import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # OpenWeatherMap sample API key (limited)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def main():
    print("Simple Weather Application")
    city = input("Enter city name: ").strip()
    if not city:
        print("City name cannot be empty.")
        return
    data = get_weather(city)
    if data and data.get("cod") == 200:
        name = data.get("name", "Unknown location")
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        print(f"Weather in {name}: {weather.capitalize()}, Temperature: {temp}°C")
    else:
        print("Could not retrieve weather data for that city.")

if __name__ == "__main__":
    main()
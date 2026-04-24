import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # OpenWeatherMap free sample API key (for demo purposes)
    base_url = "https://openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200:
            print(f"Error: {data.get('message', 'Unable to get weather data')}")
            return None
        return data
    except Exception as e:
        print("Failed to get weather data:", e)
        return None

def display_weather(data):
    if not data:
        print("No weather data to display.")
        return
    city = data.get("name")
    weather_desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    print(f"Weather in {city}:")
    print(f"Description: {weather_desc}")
    print(f"Temperature: {temp}°C")
    print(f"Feels like: {feels_like}°C")
    print(f"Humidity: {humidity}%")

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if not city:
        print("No city entered. Exiting.")
        return
    weather_data = get_weather(city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()
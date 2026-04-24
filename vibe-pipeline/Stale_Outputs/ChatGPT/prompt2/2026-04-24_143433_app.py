import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # OpenWeatherMap sample API key for testing
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None, f"Error: Unable to get weather data for {city} (status code {response.status_code})"
        data = response.json()
        if data.get("cod") != 200:
            return None, f"Error: {data.get('message', 'Unknown error')}"
        weather_info = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather_info, None
    except Exception as e:
        return None, f"Exception occurred: {e}"

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if not city:
        print("No city entered. Exiting.")
        return
    weather_info, error = get_weather(city)
    if error:
        print(error)
    else:
        print(f"Weather in {weather_info['city']}:")
        print(f"Temperature: {weather_info['temperature']}°C")
        print(f"Condition: {weather_info['description']}")
        print(f"Humidity: {weather_info['humidity']}%")
        print(f"Wind Speed: {weather_info['wind_speed']} m/s")

if __name__ == "__main__":
    main()
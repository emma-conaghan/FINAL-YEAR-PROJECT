import requests

def get_weather(city):
    api_key = "your_api_key_here"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    weather = {
        "city": data.get("name"),
        "temperature": data["main"].get("temp"),
        "description": data["weather"][0].get("description"),
        "humidity": data["main"].get("humidity"),
        "wind_speed": data["wind"].get("speed")
    }
    return weather

def main():
    print("Simple Weather App")
    city = input("Enter city name: ")
    weather = get_weather(city)
    if weather is None:
        print("Could not retrieve weather data. Please check the city name or your API key.")
    else:
        print(f"Weather in {weather['city']}:")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Condition: {weather['description']}")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Wind Speed: {weather['wind_speed']} m/s")

if __name__ == "__main__":
    main()
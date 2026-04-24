import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # sample key for openweathermap.org free API testing
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    if data.get("cod") != 200:
        return None
    weather_desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    return {
        "description": weather_desc,
        "temperature": temp,
        "feels_like": feels_like,
        "humidity": humidity
    }

def main():
    city = input("Enter city name: ").strip()
    weather = get_weather(city)
    if weather:
        print(f"Weather in {city}:")
        print(f"Description: {weather['description']}")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Feels like: {weather['feels_like']}°C")
        print(f"Humidity: {weather['humidity']}%")
    else:
        print("Could not retrieve weather data for that city.")

if __name__ == "__main__":
    main()
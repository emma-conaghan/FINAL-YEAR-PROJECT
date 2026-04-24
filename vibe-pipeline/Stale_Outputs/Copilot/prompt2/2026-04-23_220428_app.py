import requests

def get_weather(city):
    api_key = "1a8e6fb74d1d45d8902c1d8f37b7cfe0"  # Demo key from OpenWeatherMap
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Error: {data.get('message', 'Cannot get weather data')}"
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        return f"Weather in {city}:\nDescription: {weather}\nTemperature: {temp}°C\nHumidity: {humidity}%\nWind Speed: {wind} m/s"
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if not city:
        print("No city entered.")
        return
    info = get_weather(city)
    print(info)

if __name__ == "__main__":
    main()
import requests

def get_weather(city):
    api_key = "b1b15e88fa797225412429c1c50c122a1"  # sample API key for demonstration (OpenWeatherMap)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    data = get_weather(city)
    if data is None or data.get("cod") != 200:
        print("Could not retrieve weather data. Please check the city name and try again.")
        return
    print(f"Weather for {data['name']}, {data['sys']['country']}:")
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Weather: {data['weather'][0]['description'].capitalize()}")
    print(f"Humidity: {data['main']['humidity']}%")
    print(f"Wind Speed: {data['wind']['speed']} m/s")

if __name__ == "__main__":
    main()
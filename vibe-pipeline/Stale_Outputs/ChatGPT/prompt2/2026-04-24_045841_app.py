import requests

def get_weather(city):
    api_key = "your_api_key_here"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    data = get_weather(city)
    if data:
        print(f"Weather in {city}:")
        print(f"Temperature: {data['main']['temp']}°C")
        print(f"Description: {data['weather'][0]['description']}")
        print(f"Humidity: {data['main']['humidity']}%")
        print(f"Wind speed: {data['wind']['speed']} m/s")
    else:
        print("Could not retrieve weather data. Check the city name and your API key.")

if __name__ == "__main__":
    main()
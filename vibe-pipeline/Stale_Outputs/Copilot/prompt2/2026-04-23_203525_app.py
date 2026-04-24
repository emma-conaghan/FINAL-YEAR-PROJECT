import requests

def get_weather(city):
    api_key = "YOUR_OPENWEATHERMAP_API_KEY"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            print("City not found or API error.")
            return
        print(f"Weather for {city}:")
        print(f"Description: {data['weather'][0]['description']}")
        print(f"Temperature: {data['main']['temp']}°C")
        print(f"Humidity: {data['main']['humidity']}%")
        print(f"Wind Speed: {data['wind']['speed']} m/s")
    except Exception as e:
        print("Error fetching weather data:", e)

def main():
    print("Simple Weather App")
    city = input("Enter city name: ")
    get_weather(city)

if __name__ == "__main__":
    main()
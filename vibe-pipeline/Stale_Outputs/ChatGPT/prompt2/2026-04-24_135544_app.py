import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # OpenWeatherMap sample API key for testing
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return f"Error: {data.get('message', 'Cannot get weather data')}"
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        return f"Weather in {city}: {weather}, Temperature: {temp}°C, Humidity: {humidity}%"
    except Exception as e:
        return f"Error occurred: {e}"

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if not city:
        print("City name cannot be empty.")
        return
    result = get_weather(city)
    print(result)

if __name__ == "__main__":
    main()
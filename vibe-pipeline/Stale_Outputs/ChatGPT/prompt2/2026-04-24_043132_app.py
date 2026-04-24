import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # Sample API key for OpenWeatherMap's free API (demo)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            print(f"Weather in {city}: {weather_desc}")
            print(f"Temperature: {temp}°C")
            print(f"Humidity: {humidity}%")
        else:
            print(f"Error: {data.get('message', 'Cannot retrieve weather data.')}")
    except requests.RequestException as e:
        print(f"Error connecting to the weather service: {e}")

def main():
    city = input("Enter city name: ").strip()
    if city:
        get_weather(city)
    else:
        print("Please enter a valid city name.")

if __name__ == "__main__":
    main()
import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # sample public key for openweathermap.org (for demonstration)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return f"Error: {data.get('message', 'Unable to get weather data.')}"
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        return (f"Weather in {city}: {weather_desc}\n"
                f"Temperature: {temp}°C\n"
                f"Humidity: {humidity}%\n"
                f"Wind speed: {wind_speed} m/s")
    except Exception as e:
        return f"Exception occurred: {e}"

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if not city:
        print("No city entered, exiting.")
        return
    print("Getting weather data...")
    weather_info = get_weather(city)
    print(weather_info)

if __name__ == "__main__":
    main()
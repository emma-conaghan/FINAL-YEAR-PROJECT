import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # OpenWeatherMap demo API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Error: {data.get('message', 'Unknown error')}"
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        return (f"Weather in {city}:\n"
                f"Description: {weather_desc}\n"
                f"Temperature: {temp}°C\n"
                f"Humidity: {humidity}%\n"
                f"Wind Speed: {wind_speed} m/s\n")
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

def main():
    print("Simple Weather App")
    city = input("Enter a city name: ").strip()
    if city:
        print(get_weather(city))
    else:
        print("No city entered. Exiting.")

if __name__ == "__main__":
    main()
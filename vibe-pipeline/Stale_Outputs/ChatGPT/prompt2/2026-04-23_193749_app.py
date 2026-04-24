import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # OpenWeatherMap sample API key (may have limitations)
    url = f"http://openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return f"Error: {data.get('message', 'Unable to get weather')}"
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        return f"Weather in {city}:\nDescription: {weather_desc}\nTemperature: {temp}°C\nHumidity: {humidity}%"
    except Exception as e:
        return f"Exception occurred: {e}"

def main():
    city = input("Enter city name: ")
    print(get_weather(city))

if __name__ == "__main__":
    main()
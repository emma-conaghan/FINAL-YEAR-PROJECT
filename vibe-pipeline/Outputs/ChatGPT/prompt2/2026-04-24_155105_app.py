import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # Sample key from OpenWeatherMap for demo (may not work, replace with your own)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Error: {data.get('message', 'Unable to get weather data')}"
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        return f"Weather in {city}:\n- Description: {weather}\n- Temperature: {temp}°C\n- Humidity: {humidity}%"
    except Exception as e:
        return f"Exception occurred: {e}"

def main():
    print("Simple Weather App")
    city = input("Enter city name: ")
    print(get_weather(city))

if __name__ == "__main__":
    main()
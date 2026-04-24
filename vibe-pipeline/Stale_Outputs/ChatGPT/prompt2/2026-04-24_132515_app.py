import requests

def get_weather(city):
    api_key = 'YOUR_API_KEY_HERE'  # Replace with your actual OpenWeatherMap API key
    base_url = 'http://api.openweathermap.org/data/2.5/weather'
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        if response.status_code == 200:
            weather = data['weather'][0]['description']
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            print(f"Weather in {city}:")
            print(f"Description: {weather}")
            print(f"Temperature: {temp}°C")
            print(f"Humidity: {humidity}%")
            print(f"Wind speed: {wind_speed} m/s")
        else:
            print(f"Error: {data.get('message', 'Unable to get weather data')}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    print("Simple Weather Application")
    city = input("Enter the city name: ").strip()
    if city:
        get_weather(city)
    else:
        print("No city entered. Exiting.")

if __name__ == '__main__':
    main()
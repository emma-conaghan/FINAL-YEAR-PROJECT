import requests

def get_weather(city):
    api_key = 'your_api_key_here'  # Replace with your actual OpenWeatherMap API key
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}'
    response = requests.get(url)
    return response.json()

def main():
    city = input("Enter city name: ")
    data = get_weather(city)

    if data.get('cod') != 200:
        print("City not found or error fetching data.")
        return

    print(f"Weather in {data['name']}, {data['sys']['country']}:")
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Weather: {data['weather'][0]['description'].capitalize()}")
    print(f"Humidity: {data['main']['humidity']}%")
    print(f"Wind Speed: {data['wind']['speed']} m/s")

if __name__ == '__main__':
    main()
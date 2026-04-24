import requests

def get_weather(city):
    api_key = '2ed1f6da89524e72b1beb2bb65036898'
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            humidity = data['main']['humidity']
            print(f'Weather in {city}:')
            print(f'Temperature: {temp}°C')
            print(f'Description: {desc}')
            print(f'Humidity: {humidity}%')
        else:
            print(f"City '{city}' not found.")
    except Exception as e:
        print('Error retrieving weather data:', e)

def main():
    print("Weather App")
    city = input("Enter city name: ")
    get_weather(city)

if __name__ == '__main__':
    main()
import requests

def get_weather(city):
    api_key = "YOUR_API_KEY_HERE"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        main = data.get('main', {})
        weather_desc = data.get('weather', [{}])[0].get('description', 'No description')
        temp = main.get('temp', 'N/A')
        humidity = main.get('humidity', 'N/A')
        print(f"Weather in {city}:")
        print(f"Description: {weather_desc}")
        print(f"Temperature: {temp}°C")
        print(f"Humidity: {humidity}%")
    else:
        print("City not found or API error.")

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if city:
        get_weather(city)
    else:
        print("No city entered.")

if __name__ == "__main__":
    main()
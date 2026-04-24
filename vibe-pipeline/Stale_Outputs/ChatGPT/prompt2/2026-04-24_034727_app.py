import requests

def get_weather(city):
    api_key = "your_api_key_here"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def display_weather(data):
    if data:
        city = data['name']
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        print(f"Weather in {city}:\nTemperature: {temp}°C\nDescription: {desc}")
    else:
        print("Could not retrieve weather data.")

def main():
    city = input("Enter city name: ")
    data = get_weather(city)
    display_weather(data)

if __name__ == "__main__":
    main()
import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # OpenWeatherMap sample API key
    url = f"http://openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'weather' in data and 'main' in data:
            weather = data['weather'][0]['main']
            description = data['weather'][0]['description']
            temp = data['main']['temp']
            print(f"Weather in {city}:")
            print(f"{weather} - {description.capitalize()}")
            print(f"Temperature: {temp} °C")
        else:
            print("Could not find weather data for the provided city.")
    except Exception as e:
        print("Error retrieving weather data:", e)

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if city:
        get_weather(city)
    else:
        print("No city entered.")

if __name__ == "__main__":
    main()
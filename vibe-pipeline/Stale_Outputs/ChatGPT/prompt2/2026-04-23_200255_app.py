import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # OpenWeatherMap sample API key for testing
    url = f"http://openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return f"Error: {data.get('message', 'Could not retrieve weather data.')}"
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        city_name = data['name']
        country = data['sys']['country']
        return f"Weather in {city_name}, {country}: {temp}°C, {desc.capitalize()}"
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    print("Simple Weather App")
    print("Enter a city name to get current weather information.")
    while True:
        city = input("City (or 'exit' to quit): ").strip()
        if city.lower() == 'exit':
            break
        if not city:
            print("Please enter a valid city name.")
            continue
        weather = get_weather(city)
        print(weather)

if __name__ == "__main__":
    main()
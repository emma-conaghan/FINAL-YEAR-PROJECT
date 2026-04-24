import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # sample OpenWeatherMap demo API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    if data.get("cod") != 200:
        return None
    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    city_name = data["name"]
    return f"Weather in {city_name}: {weather}, Temperature: {temp}°C"

def main():
    print("Enter a city name to get the current weather:")
    city = input("> ").strip()
    result = get_weather(city)
    if result:
        print(result)
    else:
        print("Could not retrieve weather data for the city entered.")

if __name__ == "__main__":
    main()
import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # Public sample API key from OpenWeatherMap for demonstration
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None, f"Error: Unable to get weather data for {city}."
    data = response.json()
    if data.get("cod") != 200:
        return None, data.get("message", "Unknown error.")
    return data, None

def display_weather(data):
    name = data.get("name", "Unknown location")
    main = data.get("main", {})
    weather_list = data.get("weather", [{}])
    weather = weather_list[0]
    temp = main.get("temp", "N/A")
    description = weather.get("description", "N/A")
    humidity = main.get("humidity", "N/A")
    print(f"Weather for {name}:")
    print(f"Temperature: {temp}°C")
    print(f"Description: {description.capitalize()}")
    print(f"Humidity: {humidity}%")

def main():
    print("Simple Weather App")
    city = input("Enter a city name: ").strip()
    if not city:
        print("City name cannot be empty.")
        return
    data, error = get_weather(city)
    if error:
        print(error)
    else:
        display_weather(data)

if __name__ == "__main__":
    main()
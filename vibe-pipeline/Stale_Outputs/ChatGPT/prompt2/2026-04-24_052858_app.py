import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # This is a sample key for openweathermap's free testing endpoint
    url = f"http://openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Error: {data.get('message', 'Could not retrieve weather data.')}"
        weather_description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return (
            f"Weather in {city}:\n"
            f"Description: {weather_description}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%"
        )
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if not city:
        print("City name cannot be empty.")
        return
    print(get_weather(city))

if __name__ == "__main__":
    main()
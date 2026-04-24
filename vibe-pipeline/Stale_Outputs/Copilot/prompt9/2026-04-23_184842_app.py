import requests

def get_weather(city):
    api_key = "b6907d289e10d714a6e88b30761fae22"  # Sample API key from OpenWeatherMap's free demo endpoint
    url = f"https://openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            print("City not found or API error.")
            return
        temp = data["main"]["temp"]
        weather = data["weather"][0]["description"]
        print(f"Weather in {city.title()}:")
        print(f"Temperature: {temp}°C")
        print(f"Condition: {weather.capitalize()}")
    except Exception as e:
        print("Error:", e)

def main():
    print("Simple Weather App")
    city = input("Enter city name: ").strip()
    if city:
        get_weather(city)
    else:
        print("No city entered.")

if __name__ == "__main__":
    main()
import requests

def get_weather(city):
    api_key = "c5bfea7d7b1b4e679fa8d7c08cc1cb6d"  # Demo key, for OpenWeatherMap
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        if data.get("cod") != 200:
            print("City not found or error fetching data.")
            return
        temp = data["main"]["temp"]
        weather = data["weather"][0]["description"]
        print(f"Weather in {city}:")
        print(f"Temperature: {temp}°C")
        print(f"Condition: {weather}")
    except Exception as e:
        print("Error:", e)

def main():
    print("Simple Weather App")
    city = input("Enter city name: ")
    get_weather(city)

if __name__ == "__main__":
    main()
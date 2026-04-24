import requests

def get_weather(city):
    api_key = "your_api_key_here"  # Insert your API key here
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    print("Simple Weather App")
    city = input("Enter city name: ")
    weather = get_weather(city)
    if weather:
        name = weather.get("name")
        temp = weather["main"]["temp"]
        desc = weather["weather"][0]["description"]
        print(f"Weather for {name}:")
        print(f"Temperature: {temp}°C")
        print(f"Description: {desc.capitalize()}")
    else:
        print("Weather information could not be retrieved. Please check the city name or try again later.")

if __name__ == "__main__":
    main()
import requests

def get_weather(city):
    api_key = 'your_api_key_here'
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            print("City not found or API error.")
            return
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        print(f"Weather in {city}: {desc}, Temperature: {temp}°C")
    except Exception as e:
        print("Error connecting to the weather API.", e)

def main():
    print("Simple Weather App")
    city = input("Enter city name: ")
    get_weather(city)

if __name__ == "__main__":
    main()
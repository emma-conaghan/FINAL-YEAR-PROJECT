import requests

def get_weather(city):
    url = f"http://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        current = data['current_condition'][0]
        temp = current['temp_C']
        desc = current['weatherDesc'][0]['value']
        humidity = current['humidity']
        wind = current['windspeedKmph']
        return {
            "temperature": temp,
            "description": desc,
            "humidity": humidity,
            "wind_speed": wind
        }
    except Exception as e:
        return {"error": "Could not retrieve weather data."}

def main():
    print("Simple Weather App")
    city = input("Enter city name: ")
    weather = get_weather(city)
    if "error" in weather:
        print(weather["error"])
    else:
        print(f"Weather for {city}:")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Description: {weather['description']}")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Wind Speed: {weather['wind_speed']} km/h")

if __name__ == "__main__":
    main()
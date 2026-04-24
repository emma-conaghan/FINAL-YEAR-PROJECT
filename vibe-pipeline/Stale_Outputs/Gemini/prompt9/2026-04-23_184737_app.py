import urllib.request
import json

def main():
    city = input("Enter the city name: ").strip()
    if not city:
        print("Please enter a valid city name.")
        return

    try:
        # Using wttr.in which provides a JSON interface without an API key
        # Adding ?format=j1 returns a structured JSON object
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        
        request = urllib.request.Request(url)
        # Adding a User-Agent header to ensure the request is accepted
        request.add_header('User-Agent', 'Mozilla/5.0')
        
        with urllib.request.urlopen(request) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode())
                
                if 'current_condition' in data and len(data['current_condition']) > 0:
                    current = data['current_condition'][0]
                    temp_c = current.get('temp_C')
                    temp_f = current.get('temp_F')
                    weather_desc = current.get('weatherDesc', [{}])[0].get('value')
                    humidity = current.get('humidity')
                    wind_speed = current.get('windspeedKmph')

                    print(f"\nWeather report for {city.title()}:")
                    print(f"Condition: {weather_desc}")
                    print(f"Temperature: {temp_c}°C ({temp_f}°F)")
                    print(f"Humidity: {humidity}%")
                    print(f"Wind Speed: {wind_speed} km/h")
                else:
                    print("Could not find weather data for that location.")
            else:
                print("Failed to connect to the weather service.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
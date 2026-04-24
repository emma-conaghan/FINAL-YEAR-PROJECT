import urllib.request
import urllib.parse
import json

def run_weather_app():
    print("Welcome to the Simple Weather App")
    city = input("Enter the city name: ").strip()
    
    if not city:
        print("Error: City name cannot be empty.")
        return

    # Using wttr.in API which returns JSON and doesn't require an API key
    encoded_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        # Adding a User-Agent header is good practice for web requests
        headers = {'User-Agent': 'PythonWeatherScript/1.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                
                # Parse specific fields from the JSON response
                current = data['current_condition'][0]
                temp_c = current['temp_C']
                temp_f = current['temp_F']
                weather_desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind_speed = current['windspeedKmph']
                
                location = data['nearest_area'][0]
                region = location['region'][0]['value']
                country = location['country'][0]['value']

                print(f"\n--- Weather Report for {city.capitalize()} ---")
                print(f"Location: {region}, {country}")
                print(f"Condition: {weather_desc}")
                print(f"Temperature: {temp_c}°C / {temp_f}°F")
                print(f"Humidity: {humidity}%")
                print(f"Wind Speed: {wind_speed} km/h")
                print("-" * 30)
            else:
                print(f"Error: Could not retrieve data (Status code: {response.getcode()})")
                
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_weather_app()
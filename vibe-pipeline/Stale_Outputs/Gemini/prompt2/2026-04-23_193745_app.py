import urllib.request
import urllib.parse
import json
import sys

def get_weather():
    print("--- Basic Weather App ---")
    city = input("Enter the name of a city: ").strip()
    
    if not city:
        print("Error: City name cannot be empty.")
        return

    # wttr.in is a free service that does not require an API key
    # format=j1 returns data in a structured JSON format
    encoded_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    print(f"Fetching weather data for {city}...")

    try:
        # Use a User-Agent header to avoid potential blocks
        headers = {'User-Agent': 'PythonWeatherApp/1.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.getcode() != 200:
                print(f"Error: Server returned status code {response.getcode()}")
                return
                
            raw_data = response.read().decode('utf-8')
            data = json.loads(raw_data)

            # Extracting information from the wttr.in JSON structure
            current_condition = data['current_condition'][0]
            temp_c = current_condition['temp_C']
            temp_f = current_condition['temp_F']
            weather_desc = current_condition['weatherDesc'][0]['value']
            humidity = current_condition['humidity']
            wind_speed = current_condition['windspeedKmph']
            
            nearest_area = data['nearest_area'][0]
            region = nearest_area['region'][0]['value']
            country = nearest_area['country'][0]['value']

            print("\n" + "="*30)
            print(f"Weather Report: {city.upper()}")
            print(f"Location: {region}, {country}")
            print(f"Condition: {weather_desc}")
            print(f"Temperature: {temp_c}°C ({temp_f}°F)")
            print(f"Humidity: {humidity}%")
            print(f"Wind Speed: {wind_speed} km/h")
            print("="*30)

    except urllib.error.URLError as e:
        print(f"Network error: Could not connect to the weather service. {e}")
    except KeyError:
        print("Error: Could not find weather information for that location.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        get_weather()
    except KeyboardInterrupt:
        print("\nApplication closed.")
        sys.exit(0)
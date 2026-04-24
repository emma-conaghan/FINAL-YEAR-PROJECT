import urllib.request
import json
import sys

def get_weather():
    print("--- Weather Finder ---")
    try:
        city = input("Enter the name of a city: ").strip()
        if not city:
            print("City name cannot be empty.")
            return

        # wttr.in provides a JSON interface and doesn't require an API key
        # We use format=j1 for a structured JSON response
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"

        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(request) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode())
                
                if 'current_condition' not in data or not data['current_condition']:
                    print("Could not find weather data for that location.")
                    return

                current = data['current_condition'][0]
                temp_c = current['temp_C']
                temp_f = current['temp_F']
                desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind = current['windspeedKmph']
                
                location_data = data['nearest_area'][0]
                area = location_data['areaName'][0]['value']
                country = location_data['country'][0]['value']

                print(f"\nLocation: {area}, {country}")
                print(f"Condition: {desc}")
                print(f"Temperature: {temp_c}°C ({temp_f}°F)")
                print(f"Humidity: {humidity}%")
                print(f"Wind Speed: {wind} km/h")
            else:
                print(f"Error: Server returned status code {response.getcode()}")
    
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_weather()
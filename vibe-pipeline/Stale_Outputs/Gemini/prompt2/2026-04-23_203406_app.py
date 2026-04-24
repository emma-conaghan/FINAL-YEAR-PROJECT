import urllib.request
import json
import sys

def get_weather():
    print("--- Simple Weather Finder ---")
    try:
        city = input("Enter the name of a city: ").strip()
        if not city:
            print("No city entered. Exiting.")
            return

        # Using wttr.in with the format=j1 query parameter to get JSON output
        # This service is free and does not require an API key for simple requests.
        url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
        
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(request) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                
                # Parsing the nested JSON structure from wttr.in
                current = data['current_condition'][0]
                area = data['nearest_area'][0]
                
                location_name = area['areaName'][0]['value']
                country = area['country'][0]['value']
                temp_c = current['temp_C']
                weather_desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind_speed = current['windspeedKmph']

                print(f"\nResults for {location_name}, {country}:")
                print(f"Temperature: {temp_c}°C")
                print(f"Condition: {weather_desc}")
                print(f"Humidity: {humidity}%")
                print(f"Wind Speed: {wind_speed} km/h")
            else:
                print("Error: Unable to reach the weather service.")

    except urllib.error.HTTPError:
        print("Error: City not found or service unavailable.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_weather()
    input("\nPress Enter to exit...")
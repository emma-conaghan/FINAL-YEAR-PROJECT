import urllib.request
import urllib.parse
import json
import sys

def get_weather():
    city = input("Enter city name: ").strip()
    if not city:
        print("City name cannot be empty.")
        return

    # Using wttr.in JSON API (No API key required)
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())

            # Parse relevant data from wttr.in JSON format
            current = data['current_condition'][0]
            temp_c = current['temp_C']
            temp_f = current['temp_F']
            desc = current['weatherDesc'][0]['value']
            humidity = current['humidity']
            wind = current['windspeedKmph']
            
            location = data['nearest_area'][0]
            area_name = location['areaName'][0]['value']
            country = location['country'][0]['value']

            print(f"\n--- Weather Report for {area_name}, {country} ---")
            print(f"Condition:   {desc}")
            print(f"Temperature: {temp_c}°C ({temp_f}°F)")
            print(f"Humidity:    {humidity}%")
            print(f"Wind Speed:  {wind} km/h")
            print("-" * 40)

    except Exception:
        print(f"Error: Could not retrieve weather data for '{city}'.")
        print("Please check the city name and your internet connection.")

if __name__ == "__main__":
    try:
        get_weather()
    except KeyboardInterrupt:
        print("\nApplication exited.")
        sys.exit(0)
import urllib.request
import urllib.parse
import json

def fetch_weather():
    print("--- Local Weather Viewer ---")
    city = input("Enter the city name: ").strip()

    if not city:
        print("Error: City name cannot be empty.")
        return

    # Using wttr.in which provides a JSON interface without requiring an API key
    # format=j1 specifies the JSON output format
    encoded_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        # Request data using standard library to ensure it works out of the box
        req = urllib.request.Request(url, headers={'User-Agent': 'PythonWeatherApp/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                
                # Extract relevant information from the JSON structure
                current = data['current_condition'][0]
                area = data['nearest_area'][0]
                
                resolved_city = area['areaName'][0]['value']
                country = area['country'][0]['value']
                temp_c = current['temp_C']
                description = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind_speed = current['windspeedKmph']

                print(f"\nWeather results for {resolved_city}, {country}:")
                print(f"Temperature:    {temp_c}°C")
                print(f"Condition:      {description}")
                print(f"Humidity:       {humidity}%")
                print(f"Wind Speed:     {wind_speed} km/h")
            else:
                print(f"Error: Server returned status code {response.getcode()}")

    except Exception as e:
        print(f"Error: Unable to retrieve weather data for '{city}'.")
        print(f"Technical details: {e}")

if __name__ == "__main__":
    fetch_weather()
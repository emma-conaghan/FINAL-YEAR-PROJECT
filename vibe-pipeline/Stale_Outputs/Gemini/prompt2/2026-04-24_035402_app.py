import urllib.request
import urllib.parse
import json

def get_weather():
    city = input("Enter the name of a city: ").strip()
    if not city:
        print("Error: City name is required.")
        return

    # wttr.in provides a JSON output when format=j1 is specified
    # We use urllib to avoid external dependencies like 'requests'
    encoded_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        # Some services require a User-Agent header to allow programmatic access
        headers = {'User-Agent': 'PythonWeatherApp/1.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                
                # Extracting specific data from the wttr.in JSON structure
                current = data['current_condition'][0]
                temp_c = current['temp_C']
                temp_f = current['temp_F']
                desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind = current['windspeedKmph']
                
                location = data['nearest_area'][0]
                area = location['areaName'][0]['value']
                country = location['country'][0]['value']

                print(f"\nWeather Report for {area}, {country}:")
                print(f"------------------------------------")
                print(f"Condition:   {desc}")
                print(f"Temperature: {temp_c}°C ({temp_f}°F)")
                print(f"Humidity:    {humidity}%")
                print(f"Wind Speed:  {wind} km/h")
            else:
                print(f"Error: Received status code {response.getcode()}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the city name is correct and you have an internet connection.")

if __name__ == "__main__":
    get_weather()
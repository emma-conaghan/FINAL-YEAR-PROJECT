import urllib.request
import urllib.parse
import json

def fetch_weather():
    print("Simple Weather Application")
    city_name = input("Enter the name of a city: ").strip()
    
    if not city_name:
        print("Please enter a valid city name.")
        return

    # wttr.in is used as it does not require an API key and supports JSON output
    encoded_city = urllib.parse.quote(city_name)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        # User-Agent header is often required by web services
        headers = {'User-Agent': 'SimplePythonWeatherApp/1.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                raw_data = response.read().decode('utf-8')
                data = json.loads(raw_data)
                
                # Extracting relevant information from the JSON structure
                current_stats = data['current_condition'][0]
                temp_c = current_stats['temp_C']
                description = current_stats['weatherDesc'][0]['value']
                humidity_val = current_stats['humidity']
                feels_like = current_stats['FeelsLikeC']
                
                area_info = data['nearest_area'][0]
                resolved_city = area_info['areaName'][0]['value']
                country_name = area_info['country'][0]['value']
                
                print(f"\n--- Weather for {resolved_city}, {country_name} ---")
                print(f"Condition: {description}")
                print(f"Temperature: {temp_c}°C")
                print(f"Feels Like: {feels_like}°C")
                print(f"Humidity: {humidity_val}%")
            else:
                print(f"Server returned status code: {response.status}")
                
    except Exception as error_msg:
        print(f"Unable to fetch weather data: {error_msg}")

if __name__ == "__main__":
    fetch_weather()
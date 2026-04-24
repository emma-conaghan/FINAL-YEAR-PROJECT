import urllib.request
import urllib.parse
import json
import sys

def get_weather(city):
    """
    Fetches weather data from wttr.in (a free weather API that doesn't require a key).
    Uses the built-in urllib to ensure no external dependencies are needed.
    """
    try:
        # wttr.in provides a JSON format if we append ?format=j1
        encoded_city = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded_city}?format=j1"
        
        # Set a User-Agent to avoid potential blocking
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status != 200:
                print(f"Error: API returned status code {response.status}")
                return

            data = json.loads(response.read().decode('utf-8'))
            
            # Navigate the JSON structure provided by wttr.in
            current = data['current_condition'][0]
            temp_c = current['temp_C']
            temp_f = current['temp_F']
            desc = current['weatherDesc'][0]['value']
            humidity = current['humidity']
            wind = current['windspeedKmph']
            
            # Get location info from the response
            area = data['nearest_area'][0]
            region = area['region'][0]['value']
            country = area['country'][0]['value']

            print("\n" + "="*30)
            print(f"Location: {city.capitalize()} ({region}, {country})")
            print(f"Condition: {desc}")
            print(f"Temperature: {temp_c}°C / {temp_f}°F")
            print(f"Humidity: {humidity}%")
            print(f"Wind Speed: {wind} km/h")
            print("="*30)

    except Exception as e:
        print(f"\nError: Could not retrieve weather for '{city}'.")
        print(f"Details: {e}")

def main():
    print("Simple Weather Application")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            city = input("\nEnter city name: ").strip()
            
            if not city:
                continue
            
            if city.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                sys.exit(0)
                
            get_weather(city)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
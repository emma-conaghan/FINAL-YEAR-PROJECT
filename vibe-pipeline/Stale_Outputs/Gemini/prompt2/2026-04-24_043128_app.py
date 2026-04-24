import urllib.request
import urllib.parse
import sys

def get_weather():
    print("--- Weather App ---")
    city = input("Enter city name: ").strip()
    
    if not city:
        print("Error: City name cannot be empty.")
        return

    # Using wttr.in because it does not require an API key for basic usage.
    # We encode the city name to handle spaces and special characters.
    encoded_city = urllib.parse.quote(city)
    
    # URL format: format=3 shows 'City: Condition Temperature'
    url = f"https://wttr.in/{encoded_city}?format=3"

    try:
        # User-Agent header is sometimes required by wttr.in to return plain text
        request = urllib.request.Request(
            url, 
            headers={'User-Agent': 'curl/7.64.1'}
        )
        
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                weather_info = response.read().decode('utf-8').strip()
                print("\nResult:")
                print(weather_info)
            else:
                print(f"Error: Server returned status code {response.status}")
                
    except Exception as e:
        print(f"Error: Could not retrieve weather data.")
        print(f"Details: {e}")

if __name__ == "__main__":
    try:
        get_weather()
    except KeyboardInterrupt:
        print("\nApplication closed.")
        sys.exit(0)
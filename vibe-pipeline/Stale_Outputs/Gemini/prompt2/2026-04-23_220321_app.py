import urllib.request
import urllib.parse
import json

def get_weather():
    """
    Fetches weather information for a user-specified city using the wttr.in service.
    wttr.in provides a JSON interface (format=j1) that doesn't require an API key.
    """
    print("=== Local Weather Tracker ===")
    city = input("Enter the name of the city: ").strip()

    if not city:
        print("Error: City name cannot be empty.")
        return

    # URL encoding the city name to handle spaces and special characters
    safe_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{safe_city}?format=j1"

    try:
        # User-Agent header is often necessary to prevent 403 Forbidden errors from web scrapers
        request = urllib.request.Request(url, headers={'User-Agent': 'WeatherApp/1.0'})
        
        with urllib.request.urlopen(request) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                
                # Parsing wttr.in's JSON structure
                current = data['current_condition'][0]
                nearest = data['nearest_area'][0]
                
                temp_c = current['temp_C']
                temp_f = current['temp_F']
                desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind = current['windspeedKmph']
                
                found_city = nearest['areaName'][0]['value']
                region = nearest['region'][0]['value']
                country = nearest['country'][0]['value']

                print(f"\nResults for: {found_city}, {region}, {country}")
                print("-" * 30)
                print(f"Condition:    {desc}")
                print(f"Temperature:  {temp_c}°C ({temp_f}°F)")
                print(f"Humidity:     {humidity}%")
                print(f"Wind Speed:   {wind} km/h")
            else:
                print(f"Error: API returned status code {response.status}")

    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
    except KeyError:
        print("Error: Could not parse weather data. Please ensure the city name is valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_weather()
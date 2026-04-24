import urllib.request
import urllib.parse
import json
import sys

def get_weather():
    """
    Fetches weather data from wttr.in (a free weather API that doesn't require an API key).
    This script uses only the Python standard library to ensure it runs out of the box.
    """
    print("--- Simple Weather App ---")
    city = input("Enter the name of a city: ").strip()

    if not city:
        print("Error: No city name provided.")
        return

    # wttr.in returns JSON if we append ?format=j1
    # We quote the city name to handle spaces and special characters
    encoded_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        # Use a User-Agent header to avoid potential blocks on default python-urllib agents
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                raw_data = response.read().decode('utf-8')
                data = json.loads(raw_data)

                # Extracting specific data points from the wttr.in JSON structure
                current = data['current_condition'][0]
                temp_c = current['temp_C']
                temp_f = current['temp_F']
                weather_desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind_speed = current['windspeedKmph']

                print(f"\nWeather results for {city.capitalize()}:")
                print(f"Condition:    {weather_desc}")
                print(f"Temperature:  {temp_c}°C / {temp_f}°F")
                print(f"Humidity:     {humidity}%")
                print(f"Wind Speed:   {wind_speed} km/h")
            else:
                print(f"Error: Received status code {response.status}")

    except urllib.error.HTTPError as e:
        print(f"Error: Could not find weather data for '{city}'. (HTTP {e.code})")
    except urllib.error.URLError as e:
        print(f"Error: Failed to reach the server. Reason: {e.reason}")
    except (KeyError, IndexError, json.JSONDecodeError):
        print("Error: Received unexpected data format from the weather service.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        get_weather()
    except KeyboardInterrupt:
        print("\nProgram closed.")
        sys.exit(0)
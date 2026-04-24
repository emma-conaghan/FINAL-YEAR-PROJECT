import urllib.request
import urllib.parse

def main():
    """
    A simple weather application that fetches current weather for a given city
    using the wttr.in service. It requires no external libraries or API keys.
    """
    print("--- Local Weather Viewer ---")
    city = input("Enter the name of a city: ").strip()

    if not city:
        print("City name cannot be empty.")
        return

    # Using wttr.in with format=4 for a concise, human-readable output
    # format=4 includes city name, emoji, condition, and temperature
    base_url = "https://wttr.in/"
    encoded_city = urllib.parse.quote(city)
    query_params = "?format=4"
    url = base_url + encoded_city + query_params

    # Set a User-Agent to avoid potential blocks and specify terminal-like output
    headers = {"User-Agent": "curl/7.64.1"}

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                weather_info = response.read().decode("utf-8").strip()
                
                # wttr.in usually returns an error message like "Unknown location" 
                # inside a 200 response if the city isn't found.
                if "Unknown location" in weather_info or not weather_info:
                    print(f"Error: Could not find weather information for '{city}'.")
                else:
                    print("\nWeather Update:")
                    print(weather_info)
            else:
                print(f"Error: Server returned status code {response.status}")

    except urllib.error.URLError as e:
        print(f"Network error: Unable to connect to the weather service ({e.reason}).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
import urllib.request
import urllib.parse

def main():
    city = input("Enter city name: ").strip()
    if not city:
        print("City name cannot be empty.")
        return

    # Using wttr.in as it requires no API key and supports text output
    # ?0 for simplified output, ?m for metric, ?q for quiet
    url = "https://wttr.in/{}?0?m?q".format(urllib.parse.quote(city))
    
    try:
        # Using a curl-like User-Agent ensures we get plain text/ANSI formatting
        headers = {'User-Agent': 'curl/7.64.1'}
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request) as response:
            if response.getcode() == 200:
                weather_report = response.read().decode('utf-8')
                print("\n--- Current Weather for {} ---".format(city))
                print(weather_report)
            else:
                print("Failed to retrieve weather data. Status code: {}".format(response.getcode()))
    except Exception as e:
        print("An error occurred while fetching the weather: {}".format(e))

if __name__ == "__main__":
    main()
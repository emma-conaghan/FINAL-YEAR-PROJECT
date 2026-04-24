import urllib.request
import urllib.parse
import json

def main():
    city = input("Enter city name: ").strip()
    if not city:
        print("Please enter a valid city name.")
        return

    try:
        # wttr.in is a free weather service that doesn't require an API key.
        # format=j1 returns data in a structured JSON format.
        encoded_city = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded_city}?format=j1"
        
        # User-Agent header is good practice for web requests
        req = urllib.request.Request(url, headers={'User-Agent': 'PythonWeatherApp/1.0'})
        
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                raw_data = response.read().decode('utf-8')
                data = json.loads(raw_data)
                
                # Extracting values from the nested JSON response
                current = data['current_condition'][0]
                temp = current['temp_C']
                desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                feels_like = current['FeelsLikeC']
                
                location = data['nearest_area'][0]
                area = location['areaName'][0]['value']
                region = location['region'][0]['value']
                country = location['country'][0]['value']

                print("\n" + "="*40)
                print(f"Location: {area}, {region}, {country}")
                print(f"Condition: {desc}")
                print(f"Temperature: {temp}°C")
                print(f"Feels Like: {feels_like}°C")
                print(f"Humidity: {humidity}%")
                print("="*40)
            else:
                print(f"Server returned status code: {response.getcode()}")

    except Exception as e:
        print(f"Error: Could not retrieve weather data. {e}")

if __name__ == "__main__":
    main()
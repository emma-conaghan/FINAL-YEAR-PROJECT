import urllib.request
import json
import sys

def main():
    try:
        city = input("Enter city name: ").strip()
        if not city:
            print("Please enter a valid city name.")
            return

        # Using wttr.in which provides a JSON API without requiring an API key
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        
        request = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        with urllib.request.urlopen(request) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode())
                
                current = data['current_condition'][0]
                temp_c = current['temp_C']
                weather_desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind_speed = current['windspeedKmph']
                
                location = data['nearest_area'][0]
                city_found = location['areaName'][0]['value']
                country_found = location['country'][0]['value']

                print(f"\nWeather Report for {city_found}, {country_found}:")
                print("-" * 30)
                print(f"Condition:    {weather_desc}")
                print(f"Temperature:  {temp_c}°C")
                print(f"Humidity:     {humidity}%")
                print(f"Wind Speed:   {wind_speed} km/h")
                print("-" * 30)
            else:
                print("Error: Unable to retrieve weather data.")

    except Exception as e:
        print(f"Error: An error occurred while fetching data. Please ensure the city name is correct and you are connected to the internet.")

if __name__ == "__main__":
    main()
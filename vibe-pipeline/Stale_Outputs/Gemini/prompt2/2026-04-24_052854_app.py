import urllib.request
import urllib.parse
import json

def fetch_weather():
    print("=== Simple Weather App ===")
    city = input("Enter city name: ").strip()
    
    if not city:
        print("Error: City name cannot be empty.")
        return

    try:
        # Step 1: Geocoding - Convert city name to latitude and longitude
        # Using Open-Meteo's free geocoding API
        encoded_city = urllib.parse.quote(city)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
        
        with urllib.request.urlopen(geo_url) as geo_res:
            geo_data = json.loads(geo_res.read().decode())

        if "results" not in geo_data or not geo_data["results"]:
            print(f"Could not find coordinates for: {city}")
            return

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        full_name = f"{location.get('name')}, {location.get('admin1', '')} {location.get('country', '')}".strip()

        # Step 2: Weather Data - Fetch current weather for the coordinates
        # Using Open-Meteo's free weather forecast API
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        
        with urllib.request.urlopen(weather_url) as weather_res:
            weather_data = json.loads(weather_res.read().decode())

        current = weather_data.get("current_weather", {})
        temp = current.get("temperature")
        wind = current.get("windspeed")
        time = current.get("time")

        # Step 3: Display results
        print("\n" + "="*30)
        print(f"Weather Report for:")
        print(f"{full_name}")
        print("-" * 30)
        print(f"Temperature:  {temp}°C")
        print(f"Wind Speed:   {wind} km/h")
        print(f"Updated (UTC): {time}")
        print("="*30 + "\n")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_weather()
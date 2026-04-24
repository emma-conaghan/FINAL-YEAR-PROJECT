import urllib.request
import urllib.parse

def get_weather_report(target_city):
    try:
        # wttr.in provides weather via HTTP without needing an API key
        # format 1: city: condition temp
        encoded_city = urllib.parse.quote(target_city)
        request_url = f"https://wttr.in/{encoded_city}?format=1"
        
        # Using a custom User-Agent to ensure the service returns plain text
        header_data = {'User-Agent': 'curl/7.79.1'}
        web_request = urllib.request.Request(request_url, headers=header_data)
        
        with urllib.request.urlopen(web_request) as web_response:
            weather_text = web_response.read().decode('utf-8').strip()
            return weather_text
    except Exception as network_error:
        return f"Unable to fetch weather data: {network_error}"

def start_application():
    print("--- Minimal Weather Console ---")
    input_location = input("Enter city name: ")
    
    if input_location.strip():
        print("Fetching...")
        display_info = get_weather_report(input_location)
        print(f"Weather Update: {display_info}")
    else:
        print("Entry cannot be empty.")

if __name__ == "__main__":
    start_application()
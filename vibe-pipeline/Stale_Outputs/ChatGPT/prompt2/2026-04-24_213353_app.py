import tkinter as tk
import requests

API_KEY = "your_openweathermap_api_key_here"
API_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    try:
        response = requests.get(API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return f"Weather in {city}:\n{weather_desc.title()}\nTemperature: {temp}°C\nHumidity: {humidity}%"
    except Exception as e:
        return "Could not get weather information. Check the city name and your API key."

def display_weather():
    city = city_entry.get()
    if city.strip():
        weather_info = get_weather(city)
        result_label.config(text=weather_info)
    else:
        result_label.config(text="Please enter a city name.")

root = tk.Tk()
root.title("Simple Weather App")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

tk.Label(frame, text="Enter city name:").grid(row=0, column=0, sticky="w")

city_entry = tk.Entry(frame, width=30)
city_entry.grid(row=1, column=0, pady=5)

get_weather_button = tk.Button(frame, text="Get Weather", command=display_weather)
get_weather_button.grid(row=2, column=0, pady=5)

result_label = tk.Label(frame, text="", justify="left")
result_label.grid(row=3, column=0, pady=10)

root.mainloop()
import streamlit as st
import requests

st.title("Lab 5")
st.write('The "What to Wear" Bot')

def get_current_weather(location, api_key, units="imperial"):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={location}&appid={api_key}&units={units}"
    )

    response = requests.get(url)

    if response.status_code == 401:
        raise Exception("Invalid API key (401 Unauthorized). Please check and try again.")

    if response.status_code == 404:
        error_message = response.json().get("message")
        raise Exception(f"404 error: {error_message}")

    data = response.json()

    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    temp_min = data["main"]["temp_min"]
    temp_max = data["main"]["temp_max"]
    humidity = data["main"]["humidity"]

    # EXTRA “relevant data” from the lab instructions:
    description = data["weather"][0]["description"]

    return {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "humidity": round(humidity, 2),
        "description": description
    }

api_key = st.secrets["WEATHER_API"]  

location = st.text_input ("Enter a location (City, State, Country)to get the current weather and clothing recommendation:")
value = "Syracuse, NY, US"


if st.button ("Get Weather and Clothing Recommendation"):
    try:
        weather = get_current_weather(location, api_key)
        st.write(f"Current weather in {weather['location']}: {weather['description']}, {weather['temperature']}°F (feels like {weather['feels_like']}°F)")
        
        if weather["feels_like"] < 50:
            st.write("It's quite cold! Wear a heavy coat, scarf, and gloves.")
        elif 50 <= weather["feels_like"] < 70:
            st.write("It's a bit chilly. Consider wearing a light jacket or sweater.")
        else:
            st.write("The weather is warm! A t-shirt and shorts should be fine.")
    except Exception as e:
        st.error(str(e))

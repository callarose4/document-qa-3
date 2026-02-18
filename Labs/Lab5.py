import streamlit as st
import requests
from openai import OpenAI
import json


st.title("Lab 5")
st.write('The "What to Wear" Bot')

def get_current_weather(location, api_key, units="imperial"):
    # sanitize input (fixes: "USA" vs "US", trailing punctuation, extra spaces)
    location = (location or "").strip()
    location = location.replace(", USA", ", US").replace(" USA", " US")
    location = location.rstrip(" .,!;:")

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": units}

    response = requests.get(base_url, params=params)

    if response.status_code == 401:
        raise Exception("Invalid API key (401 Unauthorized). Please check and try again.")

    if response.status_code == 404:
        # show EXACT query that was sent (super helpful for debugging)
        raise Exception(f"404 error: city not found (q='{location}')")

    data = response.json()

    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    temp_min = data["main"]["temp_min"]
    temp_max = data["main"]["temp_max"]
    humidity = data["main"]["humidity"]
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
client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])  # for later use in the lab

location = st.text_input(
    "Enter a location (City, State, Country)to get the current weather and clothing recommendation:",
    value="Syracuse, NY, US"
)

tools=[
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Syracuse, NY"
                    }
                },
                "required": ["location"]
             }          
        }
    }
]

def normalize_location(loc: str) -> str:
    loc = (loc or "").strip()
    # OpenWeather expects country codes like "US" not "USA"
    loc = loc.replace(", USA", ", US").replace(" USA", " US")
    return loc

if st.button ("Get Outfit and Activity Recommendation"):
    try:
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant. You give clothing recommendations based on the current weather. "
                    "If you need weather data, call the get_current_weather tool."
                )
            }, 
            {
                "role": "user",
                "content": (
                    f"My location is {location}."
                    "What should I wear today and what outdoor activities are recommended?"
                )
            }
        ]

        first = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_message = first.choices[0].message

        # if the model asks to call the tool, we must execute it
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]

            if tool_call.function.name == "get_current_weather":
                arguments = json.loads(tool_call.function.arguments or "{}")

                tool_location = arguments.get("location") or location or "Syracuse, NY, US"
                if tool_location.count(",") == 1:  # e.g. "Syracuse, NY"
                    tool_location = tool_location + ", US"


                weather_data = get_current_weather(tool_location, api_key)

                # When tool is involved, make another call to the OpenAI API
                messages.append(assistant_message)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(weather_data)
                    }
                )

                second = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )

                final_answer = second.choices[0].message.content

                st.subheader("Weather Used")
                st.json(weather_data)

                st.subheader("Recommendation")
                st.write(final_answer)

            else:
                st.subheader("Recommendation")
                st.write(assistant_message.content)

        else:
            st.subheader("Recommendation")
            st.write(assistant_message.content)

    except Exception as e:
        st.error(f"Error: {e}")

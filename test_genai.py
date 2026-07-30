import os
from google import genai
from google.genai import types

key = "AIzaSyCpYl4tRfbWYhaDOtBRFFhLYKeG4r4drNc"
client = genai.Client(api_key=key)

try:
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents="Say hello"
    )
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Error for 2.0-flash: {e}")

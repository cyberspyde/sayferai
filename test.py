import json
import base64
import numpy as np
import requests

from utils.my_initializer import recognize_once

whisper_url = "https://cyberspyde-whisper-uz-api.hf.space/transcribe"
localhost_url = "http://localhost:7860/transcribe"

# Get the data to send in the POST request
mydata = recognize_once()
data = b"".join(mydata)

# Send the POST request
response = requests.post(localhost_url, data=data)

# Check the response
if response.status_code == 200:
    print("Transcription result:")
    print(response.text)
else:
    print("Error:", response.status_code)
import requests
import json

# Webhook URL for activating emergency
activate_emergency_url = "http://sayfer.uz:5000/notify-update"

# Prepare the request payload
payload = {
    "key": "test"
}

# Set the request headers
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
response = requests.post(activate_emergency_url, data=json.dumps(payload), headers=headers, timeout=5)

# Check the response status code
if response.status_code == 200:
    print(response.text)
else:
    print(response.text)

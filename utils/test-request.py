import requests
import json

# Webhook URL for activating emergency
activate_emergency_url = "http://127.0.0.1:5000/deactivate_emergency"

# Prepare the request payload
payload = {
    "key": "value"
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

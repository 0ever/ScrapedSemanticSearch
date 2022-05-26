import requests
import json

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
data = {'text': 'NASA'}
data_json = json.dumps(data)

resp = requests.post("http://127.0.0.1:5000/predict", data=data_json, headers=headers)

print(resp.json())
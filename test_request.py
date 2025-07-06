import requests

data = {
    "features": [14.5, 20.4, 96.3, 600.2, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2,
                 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5,
                 14.5, 20.4, 96.3, 600.2, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2]
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)

print(response.json())

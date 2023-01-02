import requests

b64 = ""
with open("api_test.b64", "r") as file:
    b64 = file.read()

json_data = {
    "input_b64": b64,
    "model_type": "dpt_beit_large_512",
    "model_path": "weights\dpt_beit_large_512.pt",
}

response = requests.post("http://127.0.0.1:8000", json=json_data)
print(response.json())
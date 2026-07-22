import requests

# declare model request url
API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
# declare model request token
API_TOKEN = "Your HuggingFace Token"

# declare the request headers
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# huggingface api request
response = requests.post(
    API_URL, headers=headers, json={"input": "你好，HuggingFace"}
)

print(response)
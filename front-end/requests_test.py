import requests

# Send a POST request to the local computer
response = requests.post('http://127.0.0.1:8080/api/data', data=b'Hello, world!')

# Check the status code of the response to ensure that the request was successful
if response.status_code == 200:
    print('The data was successfully sent to the local computer!')
else:
    print('Something went wrong.')
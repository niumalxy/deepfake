import requests
import json
import os

def test_stream_endpoint():
    url = "http://localhost:8000/api/task/stream"
    file_path = "test.png"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Sending request to {url} with {file_path}...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, stream=True)
            
            if response.status_code == 200:
                print("Connection established. Reading stream...")
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            json_data = json.loads(decoded_line)
                            print(f"Received update: {json_data}")
                        except json.JSONDecodeError:
                            print(f"Received non-JSON line: {decoded_line}")
            else:
                print(f"Error: Server returned status code {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_stream_endpoint()

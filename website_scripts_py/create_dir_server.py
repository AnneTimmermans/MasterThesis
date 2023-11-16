import requests

webdav_url = 'https://monitoring.grand-observatory.org/_webdav_'
username = 'user'
password = 'HB3218mWx74kOBV6ncJi'

# Define the path of the directory you want to create
directory_path = '/RMS/'

# Create the directory on the server
response = requests.request('MKCOL', f"{webdav_url}{directory_path}", auth=(username, password))

if response.status_code == 201:
    print("Directory created successfully.")
else:
    print(f"Failed to create the directory. Status code: {response.status_code}")

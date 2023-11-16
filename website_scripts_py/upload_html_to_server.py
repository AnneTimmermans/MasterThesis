import os
import requests

webdav_url = 'https://monitoring.grand-observatory.org/_webdav_'
username = 'user'
password = 'HB3218mWx74kOBV6ncJi'

local_folders = ['/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages', '/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/active_antennas', '/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/avg_trace_freq', '/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/bat_temp', '/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/RMS']
remote_folders = ['/', '/active_antennas/', '/avg_trace_freq/', '/bat_temp/', '/RMS/']

for i in range(len(local_folders)):
    local_folder = local_folders[i]
    remote_folder = remote_folders[i]

    try:
        # Create the remote folder if it doesn't exist
        requests.request('MKCOL', f"{webdav_url}{remote_folder}", auth=(username, password))
    except requests.exceptions.RequestException as e:
        print(f"Failed to create the remote folder: {e}")
        exit(1)

    # List all files in the local folder
    files_to_upload = [f for f in os.listdir(local_folder) if f.endswith('.html')]

    for filename in files_to_upload:
        local_path = os.path.join(local_folder, filename)
        remote_path = os.path.join(remote_folder, filename)

        try:
            with open(local_path, 'rb') as file:
                # Use data parameter to send the file's content as the request data
                response = requests.put(f"{webdav_url}{remote_path}", auth=(username, password), data=file)

            if response.status_code == 201:
                print(f"Uploaded {webdav_url}{remote_path} successfully.")
            else:
                print(f"Failed to upload {filename}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to upload {filename}: {e}")

print("HTML files uploaded.")


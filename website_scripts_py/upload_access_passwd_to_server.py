import os
import requests

webdav_url = 'https://monitoring.grand-observatory.org/_webdav_'
username = 'user'
password = 'HB3218mWx74kOBV6ncJi'

local_folders = [
    '/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/.htaccess',
    '/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/.htpasswd'
]

remote_folders = [
    '/',
    '/'
]

for i in range(len(local_folders)):
    local_folder = local_folders[i]
    remote_folder = remote_folders[i]

    try:
        # Create the remote folder if it doesn't exist
        requests.request('MKCOL', f"{webdav_url}{remote_folder}", auth=(username, password))
    except requests.exceptions.RequestException as e:
        print(f"Failed to create the remote folder: {e}")
        exit(1)

    if os.path.isfile(local_folder):
        # If it's a file, upload it
        filename = os.path.basename(local_folder)
        remote_path = os.path.join(remote_folder, filename)

        try:
            with open(local_folder, 'rb') as file:
                response = requests.put(f"{webdav_url}{remote_path}", auth=(username, password), data=file)

            if response.status_code == 201:
                print(f"Uploaded {webdav_url}{remote_path} successfully.")
            else:
                print(f"Failed to upload {filename}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to upload {filename}: {e}")

    elif os.path.isdir(local_folder):
        # If it's a directory, list all HTML files and upload them
        files_to_upload = [f for f in os.listdir(local_folder) if f.endswith('.html')]

        for filename in files_to_upload:
            local_path = os.path.join(local_folder, filename)
            remote_path = os.path.join(remote_folder, filename)

            try:
                with open(local_path, 'rb') as file:
                    response = requests.put(f"{webdav_url}{remote_path}", auth=(username, password), data=file)

                if response.status_code == 201:
                    print(f"Uploaded {webdav_url}{remote_path} successfully.")
                else:
                    print(f"Failed to upload {filename}. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to upload {filename}: {e}")

print("Files uploaded.")

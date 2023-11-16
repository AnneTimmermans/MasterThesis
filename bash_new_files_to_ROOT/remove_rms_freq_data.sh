#!/bin/bash

directory_path_rms="/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data"
directory_path_avg_freq_trace="/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data"

# Calculate the timestamp for files older than 30 days
thirty_days_ago=$(date -d '30 days ago' +%s)

# Use the find command to locate and remove old files in the first directory, and print their names
find "$directory_path_rms" -type f -mtime +30 -exec rm -v {} \;

# Use the find command to locate and remove old files in the second directory, and print their names
find "$directory_path_avg_freq_trace" -type f -mtime +30 -exec rm -v {} \;


# #!/bin/bash

# # Specify the directory paths where you want to remove old files
# directory_path_rms="/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data"
# directory_path_avg_freq_trace="/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data"

# # Calculate the timestamp for files older than 6 days
# six_days_ago=$(date -d '6 days ago' +%s)

# # Use the find command to locate and remove old files in the first directory, and print their names
# find "$directory_path_rms" -type f -mtime +6 -exec rm -v {} \;

# # Use the find command to locate and remove old files in the second directory, and print their names
# find "$directory_path_avg_freq_trace" -type f -mtime +6 -exec rm -v {} \;
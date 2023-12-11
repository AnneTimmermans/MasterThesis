#!/bin/bash

directory_path_rms="/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data"
directory_path_avg_freq_trace="/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data"

# Calculate the timestamp for files older than 30 days
thirty_days_ago=$(date -d '30 days ago' +%s)

# Use the find command to locate and remove old files in the first directory, and print their names
find "$directory_path_rms" -type f -mtime +30 -exec rm -v {} \;

# Use the find command to locate and remove old files in the second directory, and print their names
find "$directory_path_avg_freq_trace" -type f -mtime +30 -exec rm -v {} \;

# Use the find command to locate and remove old files in the third directory, and print their names
find /pbs/home/a/atimmerm/GRAND/data/YMD_GRANDfiles -type f ! -name "*.root" -exec rm {} +

# Get today's date in the format YYYYMMDD
today=$(date "+%Y/%m/%d")
Year=$(date "+%Y")

# Define source and destination directories
source_dir="/pbs/home/a/atimmerm/GRAND/data/YMD_GRANDfiles/$today"
source_dir_home="/pbs/home/a/atimmerm/GRAND/data/YMD_GRANDfiles/$Year"
dest_gaa="/sps/grand/data/gaa/GrandRoot/$(date "+%Y/%m/")"
dest_gp13="/sps/grand/data/gp13/GrandRoot/$(date "+%Y/%m/")"

# Function to create destination directories if they don't exist
create_destination_directories() {
    local dest_dir=$1
    if [ ! -d "$dest_dir" ]; then
        mkdir -p "$dest_dir"
        echo "Created directory: $dest_dir"
    fi
}

# Create destination directories if they don't exist
create_destination_directories "$dest_gaa"
create_destination_directories "$dest_gp13"

# Move files that start with 'td' from today's directory to the destination
find "$source_dir" -type f -name 'td*' -exec mv {} "$dest_gaa" \; -exec echo "Moved file: {} to $dest_gaa" \;

# Move files that start with 'GP13' from today's directory to the GP13 destination
find "$source_dir" -type f -name 'GP13*' -exec mv {} "$dest_gp13" \; -exec echo "Moved file: {} to $dest_gp13" \;

# Remove the YYYY/MM/DD folder without removing YMD_GRANDfiles
rm -r "$source_dir_home" && echo "Removed directory: $source_dir_home"

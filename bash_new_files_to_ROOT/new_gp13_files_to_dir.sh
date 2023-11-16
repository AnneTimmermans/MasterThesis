#!/bin/bash

# Define the source directory for .dat files
source_directory_dat="/sps/grand/data/gp13"

# Calculate the timestamp for yesterday at 5:00 AM
yesterday_at_5=$(date -d 'yesterday 05:00' '+%Y%m%d%H%M.%S')

# Create a reference file with the desired timestamp
touch -t "$yesterday_at_5" /tmp/last_check_reference

# Find all .dat files in the source directory that have been modified since the reference file
new_dat_files=($(find "$source_directory_dat" -type f -name "*.dat" -newer /tmp/last_check_reference))

# Define the base directory
base_directory="/sps/grand/data/auger/YMD_GRANDfiles"

# Extract Year, Month, and Day from the current date
year=$(date -d now +'%Y')
month=$(date -d now +'%m')
day=$(date -d now +'%d')

# Define the day directory
day_directory="$base_directory/$year/$month/$day"

# Create the day directory if it doesn't exist
mkdir -p "$day_directory"

# Loop through the array of new files and copy them to the day directory
for file in "${new_dat_files[@]}"; do
  cp "$file" "$day_directory/"
  echo "File $file copied to $day_directory/"
done

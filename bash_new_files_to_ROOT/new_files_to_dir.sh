#!/bin/bash

# Get the timestamp for yesterday at 5:00 AM
yesterday_at_5=$(date -d 'yesterday 05:00' '+%Y%m%d%H%M.%S')

# Get the current date in the format YYYY/MM/DD
current_date=$(date +'%Y/%m/%d')

# Define the base directory
base_directory="/sps/grand/data/auger/YMD_GRANDfiles"

# Extract Year and Month from the current date
year=$(date -d $current_date +'%Y')
month=$(date -d $current_date +'%m')

# Check if the Year directory exists, and create it if not
year_directory="$base_directory/$year"
if [ ! -d "$year_directory" ]; then
    mkdir -p "$year_directory"
fi

# Check if the Month directory exists, and create it if not
month_directory="$year_directory/$month"
if [ ! -d "$month_directory" ]; then
    mkdir -p "$month_directory"
fi

# Create the directory for the current date
day_directory="$month_directory/$(date -d $current_date +'%d')"
mkdir -p "$day_directory"

echo "Directory created: $day_directory"

# Copy new files to the Day directory
source_directory="/sps/grand/data/auger/TD"

# Calculate the timestamp for yesterday at 5:00 AM
yesterday_at_5=$(date -d 'yesterday 05:00' '+%Y%m%d%H%M.%S')

# Create a reference file with the desired timestamp
touch -t "$yesterday_at_5" /tmp/last_check_reference

# Function to process new files
process_new_files() {
  local files=("$@")
  for file in "${files[@]}"; do
    echo "New file detected: $file"
    # Add your custom logic to process the new file here
  done
}

# Find all files in the directory that have been modified since the reference file
new_files=($(find "$source_directory" -type f -newer /tmp/last_check_reference))

# Loop through the array of new files and copy them to the Day directory
for file in "${new_files[@]}"; do
  cp "$file" "$day_directory/"
  echo "File $file copied to $day_directory/"
done

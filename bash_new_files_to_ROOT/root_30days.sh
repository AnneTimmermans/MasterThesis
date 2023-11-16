#!/bin/bash

source /pbs/throng/grand/soft/miniconda3/bin/activate /sps/grand/software/conda/grandlib_2304/

# Define the base directory where your ROOT files are stored
base_directory="/sps/grand/data/auger/YMD_GRANDfiles"

# Define the target root directory where you want to store the combined files
output_root_directory="/sps/grand/data/auger/YMD_GRANDfiles"  # Update this with the desired output directory

# Get the current date in YYYY/MM/DD format
current_date=$(date +'%Y/%m/%d')

# Define the target directory for the current date
output_directory="$output_root_directory/$current_date"

# Define the date 30 days ago in YYYYMMDD format
thirty_days_ago=$(date -d "30 days ago" +%Y%m%d)

# Create a list of directories in the last 30 days (YYYY/MM/DD format)
directories_to_check=($(find "$base_directory" -type d -newermt "$thirty_days_ago" -printf "%Y/%m/%d\n" | sort -u))

# Include the current date in the list of directories
directories_to_check+=("$current_date")

# Initialize an array to store the files to combine
files_to_combine=()

# Loop through the directories and collect the files
for dir in "${directories_to_check[@]}"; do
  # Find ROOT files in the current directory
  current_directory="$base_directory/$dir"
  if [ -d "$current_directory" ]; then
    echo "Searching in directory: $current_directory"
    current_files=($(find "$current_directory" -type f -name "*.root" -print | grep -vE "30_days.root|7_days.root"))
    
    if [ ${#current_files[@]} -gt 0 ]; then
      echo "Found files in $current_directory:"
      for file in "${current_files[@]}"; do
        echo "  $file"
      done
    else
      echo "No files found in $current_directory."
    fi

    files_to_combine+=("${current_files[@]}")
  fi
done

# Check if there are files to combine
if [ ${#files_to_combine[@]} -eq 0 ]; then
  echo "No files found in the specified directories."
else
  # Create the output filename based on the current date and time
  output_filename="30_days.root"

  # Create the target directory if it doesn't exist
  mkdir -p "$output_directory"

  # Combine the files using 'hadd' and save in the target directory
  hadd "$output_directory/$output_filename" "${files_to_combine[@]}"

  echo "Combined files saved as $output_filename in $output_directory."
fi


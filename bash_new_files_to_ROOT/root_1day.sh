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

# Get the list of ".root" files in the current date directory
current_files=($(find "$base_directory/$current_date" -type f -name "*.root"))

# Check if there are files to combine
if [ ${#current_files[@]} -eq 0 ]; then
  echo "No '.root' files found in the specified directory for today."
else
  # Create the output filename based on the current date and time
  output_filename="1_day.root"

  # Create the target directory if it doesn't exist
  mkdir -p "$output_directory"

  # Combine the files using 'hadd' and save in the target directory
  hadd "$output_directory/$output_filename" "${current_files[@]}"

  echo "Combined files for today saved as $output_filename in $output_directory."
fi

#!/bin/bash

# Get the current date
current_date=$(date +'%Y/%m/%d')

# Define the base directory where the YYYY/MM/DD directory structure will be created
base_directory="/sps/grand/data/auger/YMD_GRANDfiles"

# Extract Year, Month, and Day from the current date
year=$(date -d $current_date +'%Y')
month=$(date -d $current_date +'%m')
day=$(date -d $current_date +'%d')

# Define the full directory path based on the current date
target_directory="$base_directory/$year/$month/$day"

# Use find to locate ".f****" files in a specific directory (modify the path accordingly)
find $target_directory -type f -name "*.dat" | while read -r file; do
    # Process each .f**** file here (e.g., run your gtot command)
    cd /pbs/home/a/atimmerm/GRAND/soft/grand/gtot/cmake-build-release
    gtot -g1 "$file"
    echo "File $file is converted to a ROOT file"
done

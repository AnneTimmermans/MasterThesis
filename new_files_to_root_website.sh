#!/bin/bash
# note that you should first setup the right environment
# This script to look for new files, create a YYYY/MM/DD directory, store there the binary files, convert them to ROOT

# remove old RMS and frequency data and binary files and 
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/remove_rms_freq_data.sh

# setting the environment
. /pbs/home/a/atimmerm/GRAND/monitoring_website/conda_env.sh

source /pbs/home/a/atimmerm/GRAND/soft/grand/env/setup.sh


# move websites to the right day
# . /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/new_day_page.sh

# look for new files and create directory
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/new_files_to_dir.sh
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/new_gp13_files_to_dir.sh

# convert new files to ROOT
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/binary_to_root.sh
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/binary_gp13_to_root.sh

# Remove binary files from directory
find /pbs/home/a/atimmerm/GRAND/data/YMD_GRANDfiles -type f ! -name "*.root" -exec rm {} +

# Run the Python script to create webpages
python /pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/create_webpages.py

 echo "Website pages are ready for upload"

 # Run the Python script to upload webpages
python /pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/upload_html_to_server.py

 echo "Website is ready to run!!!"


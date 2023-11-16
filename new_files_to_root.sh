#!/bin/bash
#note that you should first setup the right environment
#This script to look for new files, create a YYYY/MM/DD directory, store there the binary files, convert them to ROOT

#setting the environment
. /pbs/home/a/atimmerm/GRAND/monitoring_website/conda_env.sh

source /pbs/home/a/atimmerm/GRAND/soft/grand/env/setup.sh

#look for new files and create directory
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/new_files_to_dir.sh
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/new_gp13_files_to_dir.sh

# convert new files to ROOT
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/binary_to_root.sh
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/binary_gp13_to_root.sh

# make root file that combines all events from today
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/root_1day.sh

# make root file that combines all events from last 7 days
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/root_7days.sh

# make root file that combines all events from last 30 days
. /pbs/home/a/atimmerm/GRAND/monitoring_website/bash_new_files_to_ROOT/root_30days.sh

# Run the Python script to create webpages
python /pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/create_webpages.py

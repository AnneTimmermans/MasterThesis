import grand.dataio.root_trees as rt
import uproot
import awkward as ak
import numpy as np
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.express as px
import math
import ROOT
import os
import pandas as pd
import ast
import scipy
from datetime import timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import multiprocessing
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
import os

from grand import (
    Coordinates,
    CartesianRepresentation,
    SphericalRepresentation,
    GeodeticRepresentation,
)
from grand import ECEF, Geodetic, GRANDCS, LTP
from grand import Geomagnet

from freq_spec_trace_ch import *
from RMS_defs import *
from antenna_locations import *

# def adjust_lengths(A, B):
#     # Find the length difference
#     length_diff = len(A) - len(B)

#     # Adjust the length of B
#     if length_diff > 0:
#         # Repeat the last element of B to match the length of A
#         B_adjusted = np.concatenate([B, np.tile(B[-1], length_diff)])
#         print(len(A), len(B_adjusted))
#         print(A[:4], B_adjusted[:4])
#         return A, B_adjusted

#     elif length_diff < 0:
#         # Repeat the last element of A to match the length of B
#         A_adjusted = np.concatenate([A, np.tile(A[-1], -length_diff)])
#         print(len(B), len(A_adjusted))
#         print(B[:4], A_adjusted[:4])
#         return A_adjusted, B

#     else:
#         # No adjustment needed
#         print(len(A), len(B))
#         print(A[:4], B[:4])
#         return A, B

# Active antennas
def antennas_GA_txt(dir_str):
    dir_str = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'td')]

    GPS_LAT, GPS_LONG, GPS_ALT = antenna_locations_geodetic(dir_str)
    GRAND_X, GRAND_Y, GRAND_Z = antenna_locations_grandcs(dir_str)

    x = []
    y = []
    z = []
    labels = []
    GRAND = {}

    for du in GRAND_X.keys():
        x.append(np.mean(GRAND_X[du]))
        y.append(np.mean(GRAND_Y[du]))
        z.append(np.mean(GRAND_Z[du]))
        labels.append(du)

    GRAND['North-South [m]'] = x
    GRAND['East-West [m]'] = y
    GRAND['Height [m]'] = z
    GRAND['DUs'] = labels

    with open('GA_locations.txt','w') as data: 
        data.write(str(GRAND))
        print('created GA_locations.txt')

def active_GP13_antennas_html(dir_str, day_week_month):
    malTimeDelta = datetime.timedelta(hours=-3)
    malTZObject = datetime.timezone(malTimeDelta,
                                    name="MAL")

    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)  # List the files in the directory

    if day_week_month == '1_day':
        # Filter to only include '1_day.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'GRAND.TEST')]
    elif day_week_month == '7_days':
        # Filter to only include '7_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'7_days.root' and file.endswith(b'.root')]
    elif day_week_month == '30_days':
        # Filter to only include '30_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'30_days.root' and file.endswith(b'.root')]
    else:
        print("Invalid day_week_month value")

    if (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/GP13_locations.txt') == True):
        with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/GP13_locations.txt') as f:
            data = f.read()
            GRAND_GP13 = ast.literal_eval(data) #read as dictionary
    else:
        print("GA_locations.txt not found. Please provide the data.")

    x = GRAND_GP13['North-South [m]']
    y = GRAND_GP13['East-West [m]']
    dus = GRAND_GP13['DUs']

    # Collect active antenna positions
    x_active = []  # Initialize empty list
    y_active = []  # Initialize empty list
    du_active = []  # Initialize empty list

    x_inactive = []  # Initialize empty list
    y_inactive = []  # Initialize empty list
    du_inactive = []  # Initialize empty list

    for file in filtered_files:
        filename = os.fsdecode(file)
        fn = os.path.join(dir_str, filename)

        tadc = rt.TADC(fn)
        df = rt.DataFile(fn)
        trawv = df.trawvoltage

        count = trawv.draw('du_id', "")
        du_id = np.unique(np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(int))
        print(du_id, dus)
        index_du = [np.where(dus == du_id[i])[0][0] for i in range(len(du_id))]
        x_active.extend([x[index_du[i]] for i in range(len(index_du))])
        y_active.extend([y[index_du[i]] for i in range(len(index_du))])
        du_active.extend([du_id[i] for i in range(len(index_du))])

    # Calculate x_inactive, y_inactive, and du_inactive
    x_inactive = [x_val for x_val in x if x_val not in x_active]
    y_inactive = [y_val for y_val in y if y_val not in y_active]
    du_inactive = [du_val for du_val in dus if du_val not in du_active]

    # Create a scatter plot figure
    fig = go.Figure()

    # Add the "inactive" trace with labels
    fig.add_trace(go.Scatter(
        x=x_inactive,
        y=y_inactive,
        mode='markers',
        marker=dict(color='red'),
        name='inactive',
        text=du_inactive  # Assuming 'DUs' contains the labels for the antennas
    ))

    # Add the "active" trace with labels
    fig.add_trace(go.Scatter(
        x=x_active,
        y=y_active,
        mode='markers',
        marker=dict(color='green'),
        name='active',
        text=du_active  # Assuming 'DUs' contains the labels for active antennas
    ))

    # Set the layout properties
    fig.update_layout(title="Du's GP13")

    # Save the plot as HTML
    fig.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/active_antennas/active_GP13_antennas.html")
    print("File made active_GP13_antennas.html")

def active_GA_antennas_html(dir_str, day_week_month):

    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)  # List the files in the directory

    if day_week_month == '1_day':
        # Filter to only include '1_day.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'td')]
    elif day_week_month == '7_days':
        # Filter to only include '7_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'7_days.root' and file.endswith(b'.root')]
    elif day_week_month == '30_days':
        # Filter to only include '30_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'30_days.root' and file.endswith(b'.root')]
    else:
        print("Invalid day_week_month value")

    # Load GRAND data
    if os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/GA_locations.txt'):
        with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/GA_locations.txt') as f:
            data = f.read()
            GRAND_GA = ast.literal_eval(data)  # Read as a dictionary
    else:
        print("GA_locations.txt not found. Please provide the data.")

    x = GRAND_GA['North-South [m]']
    y = GRAND_GA['East-West [m]']
    dus = GRAND_GA['DUs']

    # Collect active antenna positions
    x_active = []  # Initialize empty list
    y_active = []  # Initialize empty list
    du_active = []  # Initialize empty list

    x_inactive = []  # Initialize empty list
    y_inactive = []  # Initialize empty list
    du_inactive = []  # Initialize empty list

    for file in filtered_files:
        filename = os.fsdecode(file)
        fn = os.path.join(dir_str, filename)

        tadc = rt.TADC(fn)
        df = rt.DataFile(fn)
        trawv = df.trawvoltage

        count = trawv.draw('du_id', "")
        du_id = np.unique(np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(int))
        # print(len(du_id), len(dus))
        print(du_id, dus)
        index_du = [np.where(dus == du_id[i])[0][0] for i in range(len(du_id))]
        x_active.extend([x[index_du[i]] for i in range(len(index_du))])
        y_active.extend([y[index_du[i]] for i in range(len(index_du))])
        du_active.extend([du_id[i] for i in range(len(index_du))])

    # Calculate x_inactive, y_inactive, and du_inactive
    x_inactive = [x_val for x_val in x if x_val not in x_active]
    y_inactive = [y_val for y_val in y if y_val not in y_active]
    du_inactive = [du_val for du_val in dus if du_val not in du_active]

    # Create a scatter plot figure
    fig = go.Figure()

    # Add the "inactive" trace with labels
    fig.add_trace(go.Scatter(
        x=x_inactive,
        y=y_inactive,
        mode='markers',
        marker=dict(color='red'),
        name='inactive',
        text=du_inactive  # Assuming 'DUs' contains the labels for the antennas
    ))

    # Add the "active" trace with labels
    fig.add_trace(go.Scatter(
        x=x_active,
        y=y_active,
        mode='markers',
        marker=dict(color='green'),
        name='active',
        text=du_active  # Assuming 'DUs' contains the labels for active antennas
    ))

    # Set the layout properties
    fig.update_layout(title="Du's GRAND@Auger")

    # Save the plot as HTML
    fig.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/active_antennas/active_GA_antennas.html")
    print("File made active_GA_antennas.html")

def active_percentage_GP13_antennas_html(dir_str, day_week_month):

    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)

    filtered_files = [file for file in root_files if file.endswith(b'.root') and file.startswith(b'GRAND.TEST')]

    if os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/GP13_locations.txt'):
        with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/GP13_locations.txt', 'r') as f:
            data = f.read()
            GRAND_GP13 = ast.literal_eval(data)
    else:
        print("GP13_locations.txt not found. Please provide the data.")
        return  # Exit the function if data is not available

    x = GRAND_GP13['North-South [m]']
    y = GRAND_GP13['East-West [m]']
    dus = GRAND_GP13['DUs']

    GRAND_GP13_active_perc = {}
    GRAND_GP13_active_perc['DUs'] = dus
    GRAND_GP13_active_perc['North-South [m]'] = x
    GRAND_GP13_active_perc['East-West [m]'] = y

    du_active_perc = np.zeros(len(dus))
    number_of_events = 0

    for file in filtered_files:
        filename = os.fsdecode(file)
        fn = os.path.join(dir_str, filename)

        tadc = rt.TADC(fn)
        df = rt.DataFile(fn)
        trawv = df.trawvoltage

        count = trawv.draw('du_id', "")
        du_id = np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(int)
        number_of_events += len(du_id)
        
        for i, du in enumerate(dus):  # Corrected the loop header to iterate over dus
            index_du = [j for j, du_id_value in enumerate(du_id) if du_id_value == du]
            du_active_perc[i] += len(index_du)
    du_active_perc = np.asarray(du_active_perc)/number_of_events
    GRAND_GP13_active_perc['active fraction DUs'] = du_active_perc

    # Create a scatter plot figure
    fig = px.scatter(GRAND_GP13_active_perc, x='North-South [m]', y='East-West [m]', title="Topology of the du's GP13",
                     color='active fraction DUs', text='DUs')

    # Set the layout properties
    fig.update_layout(title="Du's GP13")

    # Save the plot as HTML
    output_html = "/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/active_antennas/active_perc_GP13_antennas.html"
    fig.write_html(output_html)
    print("File made:", output_html)

def active_percentage_GA_antennas_html(dir_str, day_week_month):

    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)

    filtered_files = [file for file in root_files if file.endswith(b'.root') and file.startswith(b'td')]

    if os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/GA_locations.txt'):
        with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/GA_locations.txt', 'r') as f:
            data = f.read()
            GRAND_GA = ast.literal_eval(data)
    else:
        print("GA_locations.txt not found. Please provide the data.")
        return  # Exit the function if data is not available

    x = GRAND_GA['North-South [m]']
    y = GRAND_GA['East-West [m]']
    dus = GRAND_GA['DUs']

    GRAND_GA_active_perc = {}
    GRAND_GA_active_perc['DUs'] = dus
    GRAND_GA_active_perc['North-South [m]'] = x
    GRAND_GA_active_perc['East-West [m]'] = y

    du_active_perc = np.zeros(len(dus))
    number_of_events = 0

    for file in filtered_files:
        filename = os.fsdecode(file)
        fn = os.path.join(dir_str, filename)

        tadc = rt.TADC(fn)
        df = rt.DataFile(fn)
        trawv = df.trawvoltage

        count = trawv.draw('du_id', "")
        du_id = np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(int)
        number_of_events += len(du_id)

        for i, du in enumerate(dus):  # Corrected the loop header to iterate over dus
            index_du = [j for j, du_id_value in enumerate(du_id) if du_id_value == du]
            du_active_perc[i] += len(index_du)
            
    du_active_perc = np.asarray(du_active_perc)/number_of_events
    GRAND_GA_active_perc['active fraction DUs'] = du_active_perc

     # Create a scatter plot figure
    fig = px.scatter(GRAND_GA_active_perc, x='North-South [m]', y='East-West [m]', title="Topology of the du's GRAND@Auger",
                     color='active fraction DUs', text='DUs')

    # Set the layout properties
    fig.update_layout(title="Du's GA")

    # Save the plot as HTML
    output_html = "/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/active_antennas/active_perc_GA_antennas.html"
    fig.write_html(output_html)
    print("File made:", output_html)

# Battery and temperature over time
def battery_temp_time(dir_str, day_week_month, GA_or_GP13):
    """
    Input: directory string
           day_week_month = string either '1_day'/'7_days'/'30_days' --> dependent on the input data
    Output: Temperature, battery and time dictionaries with du's as keywords
    """
    malTimeDelta = datetime.timedelta(hours=-3)
    malTZObject = datetime.timezone(malTimeDelta,
                                    name="MAL")

    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)  # List the files in the directory

    if day_week_month == '1_day':
        # Filter out '7_days.root' and '30_days.root' files and check for the '.root' extension
        if GA_or_GP13 == 'GA':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'td')]
        if GA_or_GP13 == 'GP13':
                filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'GRAND.TEST')]
    elif day_week_month == '7_days':
        # Filter to only include '7_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'7_days.root' and file.endswith(b'.root')]
    elif day_week_month == '30_days':
        # Filter to only include '30_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'30_days.root' and file.endswith(b'.root')]
    else:
        print("Invalid day_week_month value")

    TEMP = {}
    BATTERY = {}
    TIME = {}
            
    for file in filtered_files:
        filename = os.fsdecode(file)
        
        fn = dir_str + filename
        TRAWV = uproot.open(fn)['trawvoltage']
        tadc  = rt.TADC(fn)
        df = rt.DataFile(fn)
        trawv = df.trawvoltage

        duid = TRAWV["du_id"].array() 
        du_list = np.unique(ak.flatten(duid))
        du_list = np.trim_zeros(du_list) #remove du 0 as it gives wrong data :(



        for du_number in du_list:#du_list:
            count = trawv.draw("du_seconds : battery_level : gps_temp","du_id == {}".format(du_number))
            trigger_time = np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(float)
            battery_level = np.array(np.frombuffer(trawv.get_v2(), count=count)).astype(float)
            temperature = np.array(np.frombuffer(trawv.get_v3(), count=count)).astype(float)


            if du_number in TEMP.keys():
                old_temp = list(TEMP[du_number])
                old_temp.extend(temperature)
                TEMP[du_number] = old_temp

                old_bat = list(BATTERY[du_number])
                old_bat.extend(battery_level)
                BATTERY[du_number] = old_bat

                old_time = list(TIME[du_number])
                old_time.extend(trigger_time)
                TIME[du_number] = old_time
            else:
                TEMP[du_number] = temperature
                BATTERY[du_number] = battery_level
                TIME[du_number] = trigger_time

    return TEMP, BATTERY, TIME
            
def bat_temp_html(dir_str, day_week_month, GA_or_GP13):
    
    """
    Input: dir_str = directory string
           day_week_month = string either '1_day'/'7_days'/'30_days' --> dependent on the input data
    Output: Webpages for battery level and temperature for each du
    """
    malTimeDelta = datetime.timedelta(hours=-3)
    malTZObject = datetime.timezone(malTimeDelta,
                                    name="MAL")
    try:
        TEMP, BATTERY, TIME = battery_temp_time(dir_str, day_week_month, GA_or_GP13)
    except:
        TEMP = {}
        BATTERY = {}
        TIME = {}


    # Initialize the subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    # Define a list of colors for DUs
    du_colors = [[247,64,73], [214,62,202], [50, 205, 50], [247,186,64], [2,207,214]] 

    for i, du_number in enumerate(TEMP.keys()):
        # Assign a color based on DU number
        color = du_colors[i % len(du_colors)]

        # Extract data for the current DU number
        temp_du = TEMP[du_number]
        time_du = TIME[du_number]
        bat_du = BATTERY[du_number]

        times_sort = np.argsort(time_du)
        time_du = np.array(time_du)[times_sort]
        temp_du = np.array(temp_du)[times_sort]
        bat_du = np.array(bat_du)[times_sort]

        times_du = []
        for t in time_du:
            times_du.append(datetime.datetime.fromtimestamp(t))

        #make average over 30 min
        if times_du != []:
            start_dt = min(times_du)
            end_dt = max(times_du)
            # difference between current and previous date
            delta = timedelta(minutes = 10)

            # store the dates between two dates in a list
            intervals = []

            while start_dt <= end_dt:
                # add current date to list by converting  it to iso format
                intervals.append(start_dt)
                # increment start date by timedelta
                start_dt += delta
            
            mean_temp_du = []
            mean_bat_du = []
            
            for i in range(len(intervals)):
                points_time = np.where(np.logical_and(np.array(times_du)>=intervals[i-1], np.array(times_du)<=intervals[i]))
                mean_bat_du.append(np.mean(bat_du[points_time]))
                mean_temp_du.append(np.mean(temp_du[points_time]))
            
        

            # Create a subplot for each DU
            trace0 = go.Scatter(x=intervals, y=mean_bat_du, mode="markers", name=f"Battery level du {du_number}", line=dict(color=f'rgba({color[0]}, {color[1]}, {color[2]}, 0.9)', width=2))
            trace1 = go.Scatter(x=intervals, y=mean_temp_du, mode="markers", name=f"Temperature du {du_number}", line=dict(color=f'rgba({color[0]}, {color[1]}, {color[2]}, 0.9)', width=2))

            fig.add_trace(trace0, row=1, col=1)
            fig.add_trace(trace1, row=2, col=1)

            # Add a title for each subplot
            fig.update_yaxes(title_text="[V]", row=1, col=1)
            fig.update_yaxes(title_text="[°C]", row=2, col=1)
            fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
            fig.update_xaxes(showticklabels=True, row=2, col=1)  # Hide x-axis labels

    # Update the layout
    fig.update_layout(
        title_text=f"Battery level and temperature for {day_week_month} at {GA_or_GP13}",
        showlegend=True,
    )
    # Save to HTML file
    fig.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/bat_temp/bat_temp_{}_{}.html".format(GA_or_GP13,day_week_month))

    print("bat_temp_{}_{}.html is created".format(GA_or_GP13,day_week_month))

# Average frequency and trace in parallel
def freq_trace(file, GA_or_GP13, Filter=True):
    '''
    Input: 
            file = file name of the root file
            filter = boolean to determine whether the events are filtered or not
            
    Output: 
            FREQ = {DU: [[avg_freq_x],[avg_freq_y],[avg_freq_z]]} 
            TRACE = {DU: [[avg_trace_x],[avg_trace_y],[avg_trace_z]]} 
            
    ''' 
    trace_file_path = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5])
    freq_file_path = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5])
    
    if os.path.exists(trace_file_path) and os.path.exists(freq_file_path):
        print("Files already exist:", trace_file_path, freq_file_path)

    else:
        FREQ, TRACE = avg_freq_trace(file, GA_or_GP13, Filter=True)
        np.savez(trace_file_path, **TRACE)
        np.savez(freq_file_path, **FREQ)
        print("Files created:", trace_file_path, freq_file_path)

def freq_trace_parallel(dir_str, day_week_month, GA_or_GP13):
    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)  # List the files in the directory

    if day_week_month == '1_day':
        # Filter out '7_days.root' and '30_days.root' files and check for the '.root' extension
        if GA_or_GP13 == 'GA':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'td')]
        if GA_or_GP13 == 'GP13':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'GRAND.TEST')]
    elif day_week_month == '7_days':
        # Filter to only include '7_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'7_days.root' and file.endswith(b'.root')]
    elif day_week_month == '30_days':
        # Filter to only include '30_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'30_days.root' and file.endswith(b'.root')]
    else:
        print("Invalid day_week_month value")

    # Create a partial function with the GA_or_GP13 argument fixed
    partial_freq_trace = partial(freq_trace, GA_or_GP13=GA_or_GP13)

    if filtered_files:
        # Construct the list of absolute file paths
        files_to_process = [os.path.abspath(os.path.join(dir_str, file.decode('utf-8'))) for file in filtered_files]
        
        # Define the number of processes to use (usually the number of CPU cores)
        num_processes = multiprocessing.cpu_count()
        print(num_processes)

        # Create a multiprocessing Pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process the files concurrently and capture the results
            results = pool.map(partial_freq_trace, files_to_process)
    
    # return results
       
def freq_trace_to_npz(dir_str, day_week_month, GA_or_GP13):

    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)  # List the files in the directory

    if day_week_month == '1_day':
        # Filter out '7_days.root' and '30_days.root' files and check for the '.root' extension
        if GA_or_GP13 == 'GA':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'td')]
        if GA_or_GP13 == 'GP13':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'GRAND.TEST')]
        

        today = f"{dir_str[-11:-7]}_{dir_str[-6:-4]}_{dir_str[-3:-1]}"#datetime.datetime.now().strftime('%Y_%m_%d')
        trace_file_path = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace_{}.npz'.format(today, GA_or_GP13)
        freq_file_path = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq_{}.npz'.format(today, GA_or_GP13)
        
        if os.path.exists(trace_file_path) and os.path.exists(freq_file_path):
            # results = freq_trace_parallel(dir_str, day_week_month, GA_or_GP13)
            print("Files already exist:", trace_file_path, freq_file_path)
            TRACE = np.load(trace_file_path, allow_pickle=True)
            FREQ = np.load(freq_file_path, allow_pickle=True)
            
        else:
            freq_trace_parallel(dir_str, day_week_month, GA_or_GP13)

            FREQ = {}
            TRACE = {}
            print("Files do not exist, creating new ones...")
            for file in filtered_files:
                filename = os.fsdecode(file)
                file = dir_str + filename

                trace_path_file = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5])
                freq_path_file = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5])
        
                if os.path.exists(trace_path_file) and os.path.exists(freq_path_file):
                    TRACE_file = np.load(trace_path_file, allow_pickle=True)
                    FREQ_file = np.load(freq_path_file, allow_pickle=True)
            
                    for du_number in FREQ_file.keys():
                        DU_freq = FREQ_file[du_number]
                        DU_trace = TRACE_file[du_number]
                        print(DU_trace)

                        if du_number in FREQ.keys():
                            old_freq = list(FREQ[du_number])
                            old_freq.extend(DU_freq)
                            FREQ[du_number] = old_freq

                            old_trace = list(TRACE[du_number])
                            old_trace.extend(DU_trace)
                            TRACE[du_number] = old_trace

                        else:
                            FREQ[du_number] = DU_freq
                            TRACE[du_number] = DU_trace
                            


            np.savez(trace_file_path, **TRACE)
            np.savez(freq_file_path, **FREQ)
            print("New files created:", trace_file_path, freq_file_path)

        return FREQ, TRACE

    elif day_week_month == '7_days':
        FREQ_7 = {}
        TRACE_7 = {}

        # Get today's date
        today = datetime.datetime.now().date()

        # Generate a list of the last 30 days in 'YYYY_MM_DD' format
        dates = [(today - timedelta(days=i)).strftime('%Y_%m_%d') for i in range(7)]

        # Generate the file paths for each date in the last 30 days
        trace_file_paths = ['/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace_{}.npz'.format(date, GA_or_GP13) for date in dates]
        freq_file_paths = ['/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq_{}.npz'.format(date, GA_or_GP13) for date in dates]
        
        for i in range(7):
            trace_file_path = trace_file_paths[i]
            freq_file_path = freq_file_paths[i]
            if os.path.exists(trace_file_path) and os.path.exists(freq_file_path):
                # results = freq_trace_parallel(dir_str, day_week_month, GA_or_GP13)
                print("Files already exist:", trace_file_path, freq_file_path)
                TRACE_day = np.load(trace_file_path, allow_pickle=True)
                FREQ_day = np.load(freq_file_path, allow_pickle=True)

                for du_number in FREQ_day.keys():
                    DU_freq = FREQ_day[du_number]
                    DU_trace = TRACE_day[du_number]
                    print(DU_trace)

                    if du_number in FREQ_7.keys():
                        old_freq = list(FREQ_7[du_number])
                        old_freq.extend(DU_freq)
                        FREQ_7[du_number] = np.mean(np.asarray(old_freq)[:min(len(old_freq), len(FREQ_7[du_number]))], axis=0)
                        #FREQ_7[du_number] = np.mean(np.asarray(old_freq),axis=0)#old_freq

                        old_trace = list(TRACE_7[du_number])
                        old_trace.extend(DU_trace)
                        TRACE_7[du_number] = np.mean(np.asarray(old_trace)[:min(len(old_trace), len(TRACE_7[du_number]))], axis=0)
                        #TRACE_7[du_number] = np.mean(np.asarray(old_trace),axis=0)#old_trace

                    else:
                        FREQ_7[du_number] = DU_freq
                        TRACE_7[du_number] = DU_trace
        
        return FREQ_7, TRACE_7
        
    elif day_week_month == '30_days':
        FREQ_30 = {}
        TRACE_30 = {}

        # Get today's date
        today = datetime.datetime.now().date()

        # Generate a list of the last 30 days in 'YYYY_MM_DD' format
        dates = [(today - timedelta(days=i)).strftime('%Y_%m_%d') for i in range(30)]

        # Generate the file paths for each date in the last 30 days
        trace_file_paths = ['/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace_{}.npz'.format(date, GA_or_GP13) for date in dates]
        freq_file_paths = ['/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq_{}.npz'.format(date, GA_or_GP13) for date in dates]
        
        for i in range(7):
            trace_file_path = trace_file_paths[i]
            freq_file_path = freq_file_paths[i]
            if os.path.exists(trace_file_path) and os.path.exists(freq_file_path):
                # results = freq_trace_parallel(dir_str, day_week_month, GA_or_GP13)
                print("Files already exist:", trace_file_path, freq_file_path)
                TRACE_day = np.load(trace_file_path, allow_pickle=True)
                FREQ_day = np.load(freq_file_path, allow_pickle=True)

                for du_number in FREQ_day.keys():
                    DU_freq = FREQ_day[du_number]
                    DU_trace = TRACE_day[du_number]
                    print(DU_trace)

                    if du_number in FREQ_30.keys():
                        old_freq = list(FREQ_30[du_number])
                        old_freq.extend(DU_freq)
                        FREQ_30[du_number] = np.mean(np.asarray(old_freq)[:min(len(old_freq), len(FREQ_30[du_number]))], axis=0)
                        #FREQ_30[du_number] = np.mean(np.asarray(old_freq),axis=0)#old_freq

                        old_trace = list(TRACE_30[du_number])
                        old_trace.extend(DU_trace)
                        TRACE_30[du_number] = np.mean(np.asarray(old_trace)[:min(len(old_trace), len(TRACE_30[du_number]))], axis=0)
                        #TRACE_30[du_number] = np.mean(np.asarray(old_trace),axis=0)#old_trace

                    else:
                        FREQ_30[du_number] = DU_freq
                        TRACE_30[du_number] = DU_trace
        
        return FREQ_30, TRACE_30
        
    else:
        print("Invalid day_week_month value")
            
def avg_freq_trace_HTML(dir_str, day_week_month, GA_or_GP13):

    FREQ, TRACE = freq_trace_to_npz(dir_str, day_week_month, GA_or_GP13)

    # Extract data for the current DU number
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.02)
    
    # Define a list of colors for DUs
    du_colors = [[247,64,73], [214,62,202], [50, 205, 50], [247,186,64], [2,207,214]] 

    for i, du_number in enumerate(TRACE.keys()):
        if not (np.shape(TRACE[du_number]) == (3,) or np.shape(TRACE[du_number][0]) == () or np.shape(TRACE[du_number][1]) == () or np.shape(TRACE[du_number][2]) == ()):
            # Assign a color based on DU number
            color = du_colors[i % len(du_colors)]

            print(np.shape(TRACE[du_number]))
            for ch in range(3):
                print(np.shape(TRACE[du_number][ch]))
                # Create line plots instead of markers
                sample_freq = 500  # [MHz]
                print(TRACE[du_number][ch])
                n_samples = len(TRACE[du_number][ch])
                fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]

                if len(fft_freq) == len(FREQ[du_number][ch]):
                    # Use the DU-specific label for all traces of the same DU
                    trace0 = go.Scattergl(x=fft_freq, y=FREQ[du_number][ch], mode="lines", name=f'freq {ch} {du_number}', line=dict(color=f'rgba({color[0]}, {color[1]}, {color[2]}, 0.5)', width=2))
                    trace1 = go.Scattergl(y=TRACE[du_number][ch], mode="lines", name=f'trace {ch} {du_number}', line=dict(color=f'rgba({color[0]}, {color[1]}, {color[2]}, 0.5)', width=2))
                    fig.add_trace(trace0, row=ch + 1, col=1)
                    fig.add_trace(trace1, row=ch + 1, col=2)

            # Configure the frequency spectrum subplot
            fig.update_xaxes(title_text='Frequency [MHz]', range=[min(fft_freq), max(fft_freq)], row=3, col=1)
            fig.update_yaxes(title_text='ch x PSD [V²/MHz]', type='log', row=1, col=1)
            fig.update_yaxes(title_text='ch y PSD [V²/MHz]', type='log', row=2, col=1)
            fig.update_yaxes(title_text='ch z PSD [V²/MHz]', type='log', row=3, col=1)

            # Configure the time average traces subplot
            fig.update_xaxes(title_text='t [ns]', range=[0, len(TRACE[du_number][ch])], row=3, col=2)
            fig.update_yaxes(title_text='ch x [ADC]', row=1, col=2)
            fig.update_yaxes(title_text='ch y [ADC]', row=2, col=2)
            fig.update_yaxes(title_text='ch z [ADC]', row=3, col=2)

    # Update the layout
    fig.update_layout(
        title_text=f"Average frequency spectrum and average trace for {day_week_month} at {GA_or_GP13}",
        showlegend=True,
    )

    # Save to HTML file
    fig.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/avg_trace_freq/avg_trace_freq_{}_{}.html".format(GA_or_GP13, day_week_month))

    print("avg_trace_freq_{}_{}.html is created".format(GA_or_GP13, day_week_month))

# RMS over time in parallel

def RMS_file(fn, GA_or_GP13):
    if GA_or_GP13 == 'GA':
        ch1 = 0
        ch2 = 1
        ch3 = 2

    if GA_or_GP13 == 'GP13':
        ch1 = 1
        ch2 = 2
        ch3 = 3

    if (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt'.format(fn[-19:-5])) == False):
        try:
            print('creating /pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt \n'.format(fn[-19:-5]))
            TRAWV = uproot.open(fn)['trawvoltage']
            TADC = uproot.open(fn)['tadc']
            tadc  = rt.TADC(fn)
            df = rt.DataFile(fn)
            trawv = df.trawvoltage

            duid = TRAWV["du_id"].array() 
            du_list = np.unique(ak.flatten(duid))
            du_list = np.trim_zeros(du_list) #remove du 0 as it gives wrong data :(

            # get the traces using uproot

            traces_array_trawv = TRAWV["trace_ch"].array()  # get the traces array
            traces_array = TADC["trace_ch"].array()  # get the traces array

            RMS = {}

            for du_number in du_list:
                count = trawv.draw("du_seconds : battery_level","du_id == {}".format(du_number))
                trigger_time = np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(float)
                battery_level = np.array(np.frombuffer(trawv.get_v2(), count=count)).astype(float)

                idx_du = duid == du_number  # creates a boolean ak array to know if/where the du_number is in the events
                idx_dupresent = ak.where(ak.sum(idx_du, axis=1)) # this computes whether the du is present in the event

                traces_array_du = traces_array[idx_du]  # gets the traces of the correct du

                result = traces_array_du[idx_dupresent] # removes the events where the DU is not present

                try:
                    new_traces = result[:, 0, ch1:ch3+1].to_numpy() # now results should be result and can be "numpied"

                    rms_du = []
                    for evt in range(0, len(new_traces)):
                        rms_ch = [trigger_time[evt], battery_level[evt]]
                        for ch in range(3):
                            rms_ch.append(RMSE(new_traces[evt][ch]))
                        rms_du.append(rms_ch)

                    if du_number in RMS.keys():
                        old_rms = RMS[du_number]
                        old_rms.extend(rms_du)
                        RMS[du_number] = old_rms
                    else:
                        RMS[du_number] = rms_du
                        
                except Exception as e:
                    print(f"Error processing du_number {du_number}: {e}")
        except:
            RMS = {}

        #Puts the rms data to a txt file
        with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt'.format(fn[-19:-5]),'w') as data: 
            data.write(str(RMS))
            print('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt created \n'.format(fn[-19:-5]))

    else:
        print('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt exists \n'.format(fn[-19:-5]))
        
def RMS_txt_parallel(dir_str, day_week_month, GA_or_GP13):
    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)  # List the files in the directory

    if day_week_month == '1_day':
        # Filter out '7_days.root' and '30_days.root' files and check for the '.root' extension
        if GA_or_GP13 == 'GA':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'td')]
        if GA_or_GP13 == 'GP13':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'GRAND.TEST')]
    elif day_week_month == '7_days':
        # Filter to only include '7_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'7_days.root' and file.endswith(b'.root')]
    elif day_week_month == '30_days':
        # Filter to only include '30_days.root' and check for the '.root' extension
        filtered_files = [file for file in root_files if file == b'30_days.root' and file.endswith(b'.root')]
    else:
        print("Invalid day_week_month value")

    # Create a partial function with the GA_or_GP13 argument fixed
    partial_RMS_file = partial(RMS_file, GA_or_GP13=GA_or_GP13)

    if filtered_files:
        # Construct the list of absolute file paths
        files_to_process = [os.path.abspath(os.path.join(dir_str, file.decode('utf-8'))) for file in filtered_files]
        
        # Define the number of processes to use (usually the number of CPU cores)
        num_processes = multiprocessing.cpu_count()
        print(num_processes)

        # Create a multiprocessing Pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process the files concurrently and capture the results
            results = pool.map(partial_RMS_file, files_to_process)

def RMS_from_txt(dir_str, day_week_month, GA_or_GP13):
    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)  # List the files in the directory
    
    
    if day_week_month == '1_day':
        # Filter out '7_days.root' and '30_days.root' files and check for the '.root' extension
        if GA_or_GP13 == 'GA':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'td')]
        if GA_or_GP13 == 'GP13':
            filtered_files = [file for file in root_files if file not in [b'7_days.root', b'30_days.root'] and file.endswith(b'.root') and file.startswith(b'GRAND.TEST')]
    
    
        today = f"{dir_str[-11:-7]}_{dir_str[-6:-4]}_{dir_str[-3:-1]}"#datetime.datetime.now().strftime('%Y_%m_%d')
        rms_file_path = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}_rms_{}.txt'.format(today, GA_or_GP13)
        
        if os.path.exists(rms_file_path):
            with open(rms_file_path) as f:
                    data = f.read()
                    RMS = ast.literal_eval(data)
                    print("File already exist:", rms_file_path)
        else:
            # create RMS files if nescessary
            RMS_txt_parallel(dir_str, day_week_month, GA_or_GP13)
            RMS = {}
            for file in filtered_files:
                filename = os.fsdecode(file)

                if (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt'.format(filename[-19:-5])) == True):

                    with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt'.format(filename[-19:-5])) as f:
                        data = f.read()
                        try:
                            RMS_file = ast.literal_eval(data) #read as dictionary
                        except:
                            RMS_file = {}

                        for du_number in RMS_file.keys():
                            rms = RMS_file[du_number]
                            if du_number in RMS.keys():
                                old_rms = RMS[du_number]
                                old_rms.extend(rms)
                                RMS[du_number] = old_rms
                            else:
                                RMS[du_number] = rms

            
            with open(rms_file_path, 'w') as data: 
                data.write(str(RMS))
                print('{} created \n'.format(rms_file_path)) 

            for du_number in RMS.keys():
                rms_file_path = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}_rms_{}_{}.txt'.format(today, GA_or_GP13, du_number)
                with open(rms_file_path, 'w') as data: 
                    data.write(str(RMS[du_number]))
                    print('{} created \n'.format(rms_file_path)) 
        return RMS
    
    elif day_week_month == '7_days':
        # Filter to only include '7_days.root' and check for the '.root' extension
        # filtered_files = [file for file in root_files if file == b'7_days.root' and file.endswith(b'.root')]
        RMS_7 = {}

        # Get today's date
        today = datetime.datetime.now().date()

        # Generate a list of the last 30 days in 'YYYY_MM_DD' format
        dates = [(today - timedelta(days=i)).strftime('%Y_%m_%d') for i in range(7)]

        # Generate the file paths for each date in the last 30 days
        rms_file_paths = ['/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}_rms_{}.txt'.format(date, GA_or_GP13) for date in dates]
        
        for i in range(7):
            rms_file_path = rms_file_paths[i]
            if os.path.exists(rms_file_path):
                
                print("Files already exist:", rms_file_path)
                with open(rms_file_path) as f:
                    data = f.read()
                    RMS_day = ast.literal_eval(data)
                for du_number in RMS_day.keys():
                    DU_rms = RMS_day[du_number]

                    if du_number in RMS_7.keys():
                        old_rms = list(RMS_7[du_number])
                        old_rms.extend(DU_rms)
                        RMS_7[du_number] = old_rms 
                    else:
                        RMS_7[du_number] = DU_rms
        
        return RMS_7

    elif day_week_month == '30_days':
        # Filter to only include '30_days.root' and check for the '.root' extension
        # filtered_files = [file for file in root_files if file == b'30_days.root' and file.endswith(b'.root')]
        RMS_30 = {}

        # Get today's date
        today = datetime.datetime.now().date()

        # Generate a list of the last 30 days in 'YYYY_MM_DD' format
        dates = [(today - timedelta(days=i)).strftime('%Y_%m_%d') for i in range(30)]

        # Generate the file paths for each date in the last 30 days
        rms_file_paths = ['/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}_rms_{}.txt'.format(date, GA_or_GP13) for date in dates]
        
        for i in range(30):
            rms_file_path = rms_file_paths[i]
            if os.path.exists(rms_file_path):
                
                print("Files already exist:", rms_file_path)
                with open(rms_file_path) as f:
                    data = f.read()
                    RMS_day = ast.literal_eval(data)
                for du_number in RMS_day.keys():
                    DU_rms = RMS_day[du_number]

                    if du_number in RMS_30.keys():
                        old_rms = list(RMS_30[du_number])
                        old_rms.extend(DU_rms)
                        RMS_30[du_number] = old_rms
                    else:
                        RMS_30[du_number] = DU_rms
        
        return RMS_30

    else:
        print("Invalid day_week_month value")

def RMS_HTML_file(file, day_week_month, GA_or_GP13):
    try:
        if GA_or_GP13 == 'GA':
            du_number = file[-7:-4] if len(file) == 99 else file[-6:-4]
        else:
            du_number = file[-8:-4]
        print(du_number, len(file))
        try:
            with open(file) as f:
                data = f.read()
                RMS = ast.literal_eval(data)
                print("File already exist:", file)
        except:
            RMS = {}

        # Initialize the subplots
        fig_lin = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        
        bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = RMS_values_filtered(RMS)
        print(len(bat_level))
        if len(bat_level)>=100000:
            print(du_number, len(bat_level))
            bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = bat_level[:80000], rms_times[:80000], rms_1[:80000], rms_2[:80000], rms_3[:80000], rms_times_1[:80000], rms_times_2[:80000], rms_times_3[:80000]
        # Average data over half hour to reduce data points
        bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = average_over_timeinterval(bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3, delta = 5)
            
        

        # Create a subplot for each DU
        trace0 = go.Scatter(x=rms_times, y=bat_level, mode="markers", name=f"Battery level du {du_number}")#, marker=dict(color=color, opacity=0.5))
        trace1 = go.Scatter(x=rms_times_1, y=rms_1, mode="markers", name=f"Ch1 du {du_number}")#, marker=dict(color=color, opacity=0.5))
        trace2 = go.Scatter(x=rms_times_2, y=rms_2, mode="markers", name=f"Ch2 du {du_number}")#, marker=dict(color=color, opacity=0.5))
        trace3 = go.Scatter(x=rms_times_3, y=rms_3, mode="markers", name=f"Ch3 du {du_number}")#, marker=dict(color=color, opacity=0.5))

        fig_lin.add_trace(trace0, row=1, col=1)
        fig_lin.add_trace(trace1, row=2, col=1)
        fig_lin.add_trace(trace2, row=3, col=1)
        fig_lin.add_trace(trace3, row=4, col=1)

        # Add a title for each subplot
        fig_lin.update_yaxes(title_text="[V]", row=1, col=1)
        fig_lin.update_yaxes(title_text="RMS [ADC]", row=2, col=1)
        fig_lin.update_yaxes(title_text="RMS [ADC]", row=3, col=1)
        fig_lin.update_yaxes(title_text="RMS [ADC]", row=4, col=1)
        fig_lin.update_xaxes(title_text="Time (UTC)", row=4, col=1)
        fig_lin.update_xaxes(showticklabels=True, row=4, col=1)  # Hide x-axis labels

        # Update the layout
        fig_lin.update_layout(
            title_text=f"RMS Traces and battery voltage for {day_week_month} at {GA_or_GP13} DU {du_number}",
            showlegend=True,
        )

        # Save to HTML file
        fig_lin.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/RMS/rms_linear_{}_{}_{}.html".format(GA_or_GP13,day_week_month, du_number))
        
        print("rms_linear_{}_{}_{}.html is created".format(GA_or_GP13,day_week_month, du_number))

    except Exception as e:
        # Log the exception
        print(f"Error in RMS_HTML_file: {e}")

def RMS_HTML_parallel(dir_str, day_week_month, GA_or_GP13):
    directory = os.fsencode(dir_str)
    root_files = os.listdir(directory)  # List the files in the directory
    today = f"{dir_str[-11:-7]}_{dir_str[-6:-4]}_{dir_str[-3:-1]}"  # datetime.datetime.now().strftime('%Y_%m_%d')

    RMS = RMS_from_txt(dir_str, day_week_month, GA_or_GP13)
    filtered_files = []
    source_html_files = []

    for du_number in RMS.keys():
        rms_file_path = '/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}_rms_{}_{}.txt'.format(today, GA_or_GP13, du_number)
        filtered_files.append(rms_file_path)
        source_html_files.append("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/RMS/rms_linear_{}_{}_{}.html".format(GA_or_GP13,day_week_month, du_number))
        print(source_html_files)

        
    # Create a partial function with the GA_or_GP13 argument fixed
    partial_RMS_HTML_file = partial(RMS_HTML_file, day_week_month=day_week_month, GA_or_GP13=GA_or_GP13)

    num_processes = multiprocessing.cpu_count()
    print(num_processes)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process the files concurrently and capture the results
        results = pool.map(partial_RMS_HTML_file, filtered_files)

def RMS_HTML(dir_str, day_week_month, GA_or_GP13):
    RMS = RMS_from_txt(dir_str, day_week_month, GA_or_GP13)
    # Initialize the subplots
    fig_lin = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    # Define a list of colors for DUs
    du_colors = ['#f74049', '#d63eca', 'limegreen', '#f7ba40', '#3ecfd6']  # You can extend this list as needed

    for i, du_number in enumerate(RMS.keys()):
        # Extract data for the current DU number
        bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = RMS_values_filtered(RMS[du_number])
        print(len(bat_level))
        if len(bat_level) >= 86400*int(day_week_month[0]):
            # 8640 times 10 sec in one day
            bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = bat_level[-86400*int(day_week_month[0]):], rms_times[-86400*int(day_week_month[0]):], rms_1[-86400*int(day_week_month[0]):], rms_2[-86400*int(day_week_month[0]):], rms_3[-86400*int(day_week_month[0]):], rms_times_1[-86400*int(day_week_month[0]):], rms_times_2[-86400*int(day_week_month[0]):], rms_times_3[-86400*int(day_week_month[0]):]
            print(len(bat_level))
            # Average data over 5 min to reduce data points
            bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = average_over_timeinterval(bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3, delta = 5)
            print(len(bat_level))
        else:
            # Average data over 5 min to reduce data points
            bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = average_over_timeinterval(bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3, delta = 5)
            print(len(bat_level))
    

        # Assign a color based on DU number
        color = du_colors[i % len(du_colors)]

        # Create a subplot for each DU
        trace0 = go.Scatter(x=rms_times, y=bat_level, mode="markers", name=f"Battery level du {du_number}", marker=dict(color=color, opacity=0.5))
        trace1 = go.Scatter(x=rms_times_1, y=rms_1, mode="markers", name=f"Ch1 du {du_number}", marker=dict(color=color, opacity=0.5))
        trace2 = go.Scatter(x=rms_times_2, y=rms_2, mode="markers", name=f"Ch2 du {du_number}", marker=dict(color=color, opacity=0.5))
        trace3 = go.Scatter(x=rms_times_3, y=rms_3, mode="markers", name=f"Ch3 du {du_number}", marker=dict(color=color, opacity=0.5))

        fig_lin.add_trace(trace0, row=1, col=1)
        fig_lin.add_trace(trace1, row=2, col=1)
        fig_lin.add_trace(trace2, row=3, col=1)
        fig_lin.add_trace(trace3, row=4, col=1)

        # Add a title for each subplot
        fig_lin.update_yaxes(title_text="[V]", row=1, col=1)
        fig_lin.update_yaxes(title_text="RMS [ADC]", row=2, col=1)
        fig_lin.update_yaxes(title_text="RMS [ADC]", row=3, col=1)
        fig_lin.update_yaxes(title_text="RMS [ADC]", row=4, col=1)
        fig_lin.update_xaxes(title_text="Time (UTC)", row=4, col=1)
        fig_lin.update_xaxes(showticklabels=True, row=4, col=1)  # Hide x-axis labels

    # Update the layout
    fig_lin.update_layout(
        title_text=f"RMS Traces and battery voltage for {day_week_month} at {GA_or_GP13}",
        showlegend=True,
    )

    # Save to HTML file
    fig_lin.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/RMS/rms_linear_{}_{}.html".format(GA_or_GP13,day_week_month))
    
    print("rms_linear_{}_{}.html is created".format(GA_or_GP13,day_week_month))




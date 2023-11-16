import grand.dataio.root_trees as rt
import uproot
import awkward as ak
import numpy as np
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import math
import ROOT
import os
import ast
import scipy
from datetime import timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

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

# Active antennas
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

    if (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/GP13_locations.txt') == True):
        with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/GP13_locations.txt') as f:
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
    malTimeDelta = datetime.timedelta(hours=-3)
    malTZObject = datetime.timezone(malTimeDelta,
                                    name="MAL")

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
    if os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/GA_locations.txt'):
        with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/python_scripts/GA_locations.txt') as f:
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
            count = trawv.draw("du_seconds : battery_level : atm_temperature","du_id == {}".format(du_number))
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

    TEMP, BATTERY, TIME = battery_temp_time(dir_str, day_week_month, GA_or_GP13)

    # Initialize the subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for i, du_number in enumerate(TEMP.keys()):
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
            times_du.append(datetime.datetime.fromtimestamp(t).astimezone(malTZObject))

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
            trace0 = go.Scatter(x=intervals, y=mean_bat_du, mode="markers", name=f"Battery level du {du_number}")
            trace1 = go.Scatter(x=intervals, y=mean_temp_du, mode="markers", name=f"Temperature du {du_number}")

            fig.add_trace(trace0, row=1, col=1)
            fig.add_trace(trace1, row=2, col=1)

            # Add a title for each subplot
            fig.update_yaxes(title_text="[V]", row=1, col=1)
            fig.update_yaxes(title_text="[°C]", row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_xaxes(showticklabels=True, row=2, col=1)  # Hide x-axis labels

    # Update the layout
    fig.update_layout(
        title_text=f"Battery level and temperature for {day_week_month} at {GA_or_GP13}",
        showlegend=True,
    )
    # Save to HTML file
    fig.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/bat_temp/bat_temp_{}_{}.html".format(GA_or_GP13,day_week_month))

    print("bat_temp_{}_{}.html is created".format(GA_or_GP13,day_week_month))

# RMS over time
def RMS_txt(dir_str, day_week_month, GA_or_GP13):
    
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

    RMS = {}

    for file in filtered_files:
        filename = os.fsdecode(file)

        if (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt'.format(filename[0:-5])) == False):
            print('creating rms_data/{}.txt'.format(filename[0:-5]))
            fn = dir_str + filename
            TRAWV = uproot.open(fn)['trawvoltage']
            tadc  = rt.TADC(fn)
            df = rt.DataFile(fn)
            trawv = df.trawvoltage

            test_array = get_test_array(trawv)

            duid = TRAWV["du_id"].array() 
            du_list = np.unique(ak.flatten(duid))
            du_list = np.trim_zeros(du_list) #remove du 0 as it gives wrong data :(

            # get the traces using uproot

            traces_array = TRAWV["trace_ch"].array()  # get the traces array

            if GA_or_GP13 == 'GA':
                ch1 = 0
                ch3 = 2

            if GA_or_GP13 == 'GP13':
                ch1 = 1
                ch3 = 3

            RMS = {}

            for du_number in du_list:#du_list:
                count = trawv.draw("du_seconds : battery_level","du_id == {}".format(du_number))
                trigger_time = np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(float)
                battery_level = np.array(np.frombuffer(trawv.get_v2(), count=count)).astype(float)

                idx_du = duid == du_number  # creates an boolean ak array to know if/where the du_number is in the events
                idx_dupresent = ak.where(ak.sum(idx_du, axis=1)) # this computes whether the du is present in the event

                traces_array_du = traces_array[idx_du]  # gets the traces of the correct du

                result = traces_array_du[idx_dupresent] # removes the events where the DU is not present

                try:
                    traces_np = result[:, 0, ch1:ch3+1].to_numpy() # now results should be result and can be "numpied"

                    new_traces, weirdos = filter_weird_events(traces_np, du_number, test_array, GA_or_GP13) # filter the traces

                    if weirdos != []:
                        weirdos_cut = np.delete(weirdos, [0,1], axis = 1)
                        trigger_time = np.delete(trigger_time, np.unique(weirdos_cut), axis = 0)
                        battery_level = np.delete(battery_level, np.unique(weirdos_cut), axis = 0)

                    rms_du = []
                    for evt in range(0,len(new_traces)):
                        rms_ch = [trigger_time[evt],battery_level[evt]]
                        for ch in range(3):
                            rms_ch.append(RMSE(new_traces[evt][ch]))
                        rms_du.append(rms_ch)



                    if du_number in RMS.keys():
                        old_rms = RMS[du_number]
                        old_rms.extend(rms_du)
                        RMS[du_number] = old_rms
                    else:
                        RMS[du_number] = rms_du

                except ValueError as e:
                    # Catch the specific error (ValueError) for inconsistent subarray lengths
                    print(f"Skipping a subarray: {e}")
                    continue


            #Puts the rms data to a txt file
            with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt'.format(filename[0:-5]),'w') as data: 
                data.write(str(RMS))
                print('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt created'.format(filename[0:-5]))

        else:
            print('rms_data/{}.txt exists'.format(filename[0:-5]))
        
    RMS = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt'.format(filename[0:-5])) == True):

            with open('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/rms_data/{}.txt'.format(filename[0:-5])) as f:
                data = f.read()
                RMS_file = ast.literal_eval(data) #read as dictionary

                for du_number in RMS_file.keys():
                    rms = RMS_file[du_number]
                    if du_number in RMS.keys():
                        old_rms = RMS[du_number]
                        old_rms.extend(rms)
                        RMS[du_number] = old_rms
                    else:
                        RMS[du_number] = rms
                        
    return RMS

def RMS_HTML(dir_str, day_week_month, GA_or_GP13):
    
    RMS = RMS_txt(dir_str, day_week_month, GA_or_GP13)
    # Initialize the subplots
    fig_lin = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for i, du_number in enumerate(RMS.keys()):
        # Extract data for the current DU number
        bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = RMS_values_filtered(RMS, du_number)
        # Average data over half hour to reduce data points
        bat_level, intervals, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = average_over_timeinterval(bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3, delta = 30)



        # Create a subplot for each DU
        trace0 = go.Scatter(x=intervals, y=bat_level, mode="markers", name=f"Battery level du {du_number}")
        trace1 = go.Scatter(x=rms_times_1, y=rms_1, mode="markers", name=f"Ch1 du {du_number}")
        trace2 = go.Scatter(x=rms_times_2, y=rms_2, mode="markers", name=f"Ch2 du {du_number}")
        trace3 = go.Scatter(x=rms_times_3, y=rms_3, mode="markers", name=f"Ch3 du {du_number}")

        fig_lin.add_trace(trace0, row=1, col=1)
        fig_lin.add_trace(trace1, row=2, col=1)
        fig_lin.add_trace(trace2, row=3, col=1)
        fig_lin.add_trace(trace3, row=4, col=1)

        # Add a title for each subplot
        fig_lin.update_yaxes(title_text="[V]", row=1, col=1)
        fig_lin.update_yaxes(title_text="RMS [μV]", row=2, col=1)
        fig_lin.update_yaxes(title_text="RMS [μV]", row=3, col=1)
        fig_lin.update_yaxes(title_text="RMS [μV]", row=4, col=1)
        fig_lin.update_xaxes(title_text="Time", row=4, col=1)
        fig_lin.update_xaxes(showticklabels=True, row=4, col=1)  # Hide x-axis labels

    # Update the layout
    fig_lin.update_layout(
        title_text=f"RMS Traces and battery voltage for {day_week_month} at {GA_or_GP13}",
        showlegend=True,
    )

    # Save to HTML file
    fig_lin.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/RMS/rms_linear_{}_{}.html".format(GA_or_GP13,day_week_month))
    
    print("rms_linear_{}_{}.html is created".format(GA_or_GP13,day_week_month))
    
    
    
    # # Initialize the subplots
    # fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # for i, du_number in enumerate(RMS.keys()):
    #     # Extract data for the current DU number
    #     bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3 = RMS_values_filtered_HMS(RMS, du_number)

    #     mean_bat_level, intervals, mean_rms_1, mean_rms_2, mean_rms_3, rms_time_1, rms_time_2, rms_time_3 = average_over_timeinterval(bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3, delta = 10)


    #      # Create a subplot for each DU
    #     trace0 = go.Scatter(x=intervals, y=mean_bat_level, mode="markers", name=f"Battery level du {du_number}")
    #     trace1 = go.Scatter(x=rms_time_1, y=mean_rms_1, mode="markers", name=f"Ch1 du {du_number}")
    #     trace2 = go.Scatter(x=rms_time_2, y=mean_rms_2, mode="markers", name=f"Ch2 du {du_number}")
    #     trace3 = go.Scatter(x=rms_time_3, y=mean_rms_3, mode="markers", name=f"Ch3 du {du_number}")

    #     fig.add_trace(trace0, row=1, col=1)
    #     fig.add_trace(trace1, row=2, col=1)
    #     fig.add_trace(trace2, row=3, col=1)
    #     fig.add_trace(trace3, row=4, col=1)

    #     # Add a title for each subplot
    #     fig.update_yaxes(title_text="[V]", row=1, col=1)
    #     fig.update_yaxes(title_text="RMS [μV]", row=2, col=1)
    #     fig.update_yaxes(title_text="RMS [μV]", row=3, col=1)
    #     fig.update_yaxes(title_text="RMS [μV]", row=4, col=1)
    #     fig.update_xaxes(title_text="Time", row=4, col=1)
    #     fig.update_xaxes(showticklabels=True, row=4, col=1)  # Hide x-axis labels

    # # Update the layout
    # fig.update_layout(
    #     title_text=f"looped RMS Traces and battery voltage",
    #     showlegend=True,
    # )

    # # Save to HTML file
    # fig.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/RMS/rms_looped_{}_{}.html".format(GA_or_GP13,day_week_month))
    
    # print("rms_looped_{}_{}.html is created".format(GA_or_GP13,day_week_month))

# Average trace and frequency spectrum
def avg_freq_trace_npz(dir_str, day_week_month, GA_or_GP13):
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

    # Now, filtered_files contains the list of root files based on the selected day_week_month value and the '.root' extension


    FREQ = {}
    TRACE = {}

    for file in filtered_files:
        filename = os.fsdecode(file)
        file = dir_str + filename

        FREQ_file, TRACE_file = freq_to_npz(file, GA_or_GP13, Filter=True)

        for du_number in FREQ_file.keys():
            DU_freq = FREQ_file[du_number]
            DU_trace = TRACE_file[du_number]

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
                
                
    return FREQ, TRACE

def avg_freq_trace_HTML(dir_str, day_week_month, GA_or_GP13):

    FREQ, TRACE = avg_freq_trace_npz(dir_str, day_week_month, GA_or_GP13)
    # Extract data for the current DU number
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.02)

    for i, du_number in enumerate(TRACE.keys()):
        if not np.shape(TRACE[du_number]) == (3,):

            for ch in range(3):
                # Create line plots instead of markers
                sample_freq = 500 # [MHz]
                n_samples = len(TRACE[du_number][ch])
                fft_freq  = np.fft.rfftfreq(n_samples) * sample_freq # [MHz]

                if len(fft_freq) == len(FREQ[du_number][ch]):
                    trace0 = go.Scattergl(x=fft_freq, y=FREQ[du_number][ch], mode="lines", name='freq ch {} {}'.format(ch, du_number))#, line=dict(color=colors[ch]))
                    trace1 = go.Scattergl(y=TRACE[du_number][ch], mode="lines", name='trace ch {} {}'.format(ch, du_number))#, line=dict(color=colors[ch]))
                    fig.add_trace(trace0, row=ch+1, col=1)
                    fig.add_trace(trace1, row=ch+1, col=2)

            # Configure the frequency spectrum subplot
            fig.update_xaxes(title_text='Frequency [MHz]', range=[min(fft_freq), max(fft_freq)], row=3, col=1)
            fig.update_yaxes(title_text='Amplitude [A.U.]', type='log', row=1, col=1)
            fig.update_yaxes(title_text='Amplitude [A.U.]', type='log', row=2, col=1)
            fig.update_yaxes(title_text='Amplitude [A.U.]', type='log', row=3, col=1)

            # Configure the time average traces subplot
            fig.update_xaxes(title_text='t [ns]', range=[0, len(TRACE[du_number][ch])], row=3, col=2)
            fig.update_yaxes(title_text='[μV]', row=1, col=2)
            fig.update_yaxes(title_text='[μV]', row=2, col=2)
            fig.update_yaxes(title_text='[μV]', row=3, col=2)

    # Update the layout
    fig.update_layout(
        title_text=f"Average frequency spectrum and trace for {day_week_month} at {GA_or_GP13}",
        showlegend=True,
    )

    # Save to HTML file
    fig.write_html("/pbs/home/a/atimmerm/GRAND/monitoring_website/webpages/avg_trace_freq/avg_trace_freq_{}_{}.html".format(GA_or_GP13,day_week_month))

    print("avg_trace_freq_{}_{}.html is created".format(GA_or_GP13,day_week_month))


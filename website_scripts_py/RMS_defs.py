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


from freq_spec_trace_ch import *

def RMSE(y_actual):
    y_predicted = np.zeros(len(y_actual))
    MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
    RMSE = math.sqrt(MSE)
    
    return RMSE

def RMS_to_txt(dir_str):
    '''
    Save RMS values to txt file
    '''
    directory = os.fsencode(dir_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if (os.path.exists('rms_data/{}.txt'.format(filename[0:-5])) == False):
            fn = dir_str + filename
            TRAWV = uproot.open(fn)['trawvoltage']
            tadc  = rt.TADC(fn)
            df = rt.DataFile(fn)
            trawv = df.trawvoltage

            duid = TRAWV["du_id"].array() 
            du_list = np.unique(ak.flatten(duid))
            du_list = np.trim_zeros(du_list) #remove du 0 as it gives wrong data :(

            # get the traces using uproot

            traces_array = TRAWV["trace_ch"].array()  # get the traces array

            RMS = {}

            for du_number in du_list:#du_list:
                count = trawv.draw("du_seconds : battery_level","du_id == {}".format(du_number))
                trigger_time = np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(float)
                battery_level = np.array(np.frombuffer(trawv.get_v2(), count=count)).astype(float)

                idx_du = duid == du_number  # creates an boolean ak array to know if/where the du_number is in the events
                idx_dupresent = ak.where(ak.sum(idx_du, axis=1)) # this computes whether the du is present in the event

                traces_array_du = traces_array[idx_du]  # gets the traces of the correct du

                result = traces_array_du[idx_dupresent] # removes the events where the DU is not present

                traces_np = result[:, 0, 0:3].to_numpy() # now results should be result and can be "numpied"

                new_traces, weirdos = filter_weird_events(traces_np, du_number) # filter the traces

                if weirdos != []:
                    weirdos_cut = np.delete(weirdos, [0,1], axis = 1)
                    trigger_time = np.delete(trigger_time, np.unique(weirdos_cut), axis = 0)
                    battery_level = np.delete(battery_level, np.unique(weirdos_cut), axis = 0)

                rms_du = []
                for evt in range(0,len(new_traces)):
                    rms_ch = [trigger_time[evt],battery_level[evt]]
                    for ch in range(3):
                        trace_0 = new_traces[evt][ch]
                        spectrum_0 = np.abs(np.fft.rfft(trace_0))
                        peaks_0 = find_peaks(spectrum_0, height = np.mean(spectrum_0) + 3 * np.std(spectrum_0))[0]
                        peaks_0 = np.append(peaks_0, np.where(np.logical_and(freq>=85, freq<=110))) # filter fm band
                        spectrum_0 = np.delete(spectrum_0, peaks_0)
                        re_trace_0 = np.fft.irfft(spectrum_0)[1:]
                        rms_ch.append(RMSE(re_trace_0))
                    rms_du.append(rms_ch)



                if du_number in RMS.keys():
                    old_rms = RMS[du_number]
                    old_rms.extend(rms_du)
                    RMS[du_number] = old_rms
                else:
                    RMS[du_number] = rms_du


            #Puts the rms data to a txt file
            with open('rms_data/{}.txt'.format(filename[0:-5]),'w') as data: 
                data.write(str(RMS))

        else:
            print('rms_data/{}.txt exists'.format(filename[0:-5])) 

def RMS_from_txt(dir_str):
    '''
    Read RMS values from txt file
    '''
    directory = os.fsencode(dir_str)
    RMS = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if (os.path.exists('rms_data/{}.txt'.format(filename[0:-5])) == True):

            with open('rms_data/{}.txt'.format(filename[0:-5])) as f:
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

def make_or_read_all_RMS(dir_str):
    
    directory = os.fsencode(dir_str)
    
    if (os.path.exists('rms_data/all_RMS.txt') == False):
        RMS = {}
        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            if (os.path.exists('rms_data/{}.txt'.format(filename[0:-5])) == True):

                with open('rms_data/{}.txt'.format(filename[0:-5])) as f:
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


        with open('rms_data/all_RMS.txt','w') as data: 
                    data.write(str(RMS))

    elif (os.path.exists('rms_data/all_RMS.txt') == True):
        with open('rms_data/all_RMS.txt') as f:
            data = f.read()
            RMS = ast.literal_eval(data) #read as dictionary
    
    return RMS

def RMS_values_filtered(RMS):
    '''
    returns filtered rms values and trigger times
    '''

    rms_x = RMS
    rms_time = [item[0] for item in rms_x]
    bat_level = [item[1] for item in rms_x]
    rms_1 = np.asarray([item[2] for item in rms_x])
    rms_2 = np.asarray([item[3] for item in rms_x])
    rms_3 = np.asarray([item[4] for item in rms_x])
    
    times_sort = np.argsort(rms_time)
    rms_time = np.array(rms_time)[times_sort]
    bat_level = np.array(bat_level)[times_sort]
    rms_1 = np.array(rms_1)[times_sort]
    rms_2 = np.array(rms_2)[times_sort]
    rms_3 = np.array(rms_3)[times_sort]
    
    
    rms_times = []
    for t in rms_time:
        rms_times.append(datetime.datetime.fromtimestamp(t))
       
    rms_times_1 = []
    for t in rms_time:#rms_time_1:
        rms_times_1.append(datetime.datetime.fromtimestamp(t))
        
    rms_times_2 = []
    for t in rms_time:#rms_time_2:
        rms_times_2.append(datetime.datetime.fromtimestamp(t))
        
    rms_times_3 = []
    for t in rms_time:#rms_time_3:
        rms_times_3.append(datetime.datetime.fromtimestamp(t))
        
    
        
    return bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3

def RMS_values_filtered_HMS(RMS, du_number):
    '''
    returns filtered rms values and trigger times
    '''
    malTimeDelta = datetime.timedelta(hours=-3)
    malTZObject = datetime.timezone(malTimeDelta,
                                name="MAL")
    rms_x = RMS[du_number]
    rms_time = [item[0] for item in rms_x]
    bat_level = [item[1] for item in rms_x]
    rms_1 = np.asarray([item[2] for item in rms_x])
    rms_2 = np.asarray([item[3] for item in rms_x])
    rms_3 = np.asarray([item[4] for item in rms_x])
    
    rms_1_over = np.where(rms_1 >= np.mean(rms_1) + 0.2*np.mean(rms_1))
    rms_1 = np.delete(rms_1, rms_1_over, axis = 0)
    rms_time_1 = np.delete(rms_time, rms_1_over, axis = 0)
    index, peaks = scipy.signal.find_peaks(rms_1)
    rms_1 = np.delete(rms_1, index, axis = 0)
    rms_time_1 = np.delete(rms_time_1, index, axis = 0)
    
    rms_2_over = np.where(rms_2 >= np.mean(rms_2) + 0.2*np.mean(rms_2))
    rms_2 = np.delete(rms_2, rms_2_over, axis = 0)
    rms_time_2 = np.delete(rms_time, rms_2_over, axis = 0)
    index, peaks = scipy.signal.find_peaks(rms_2)
    rms_2 = np.delete(rms_2, index, axis = 0)
    rms_time_2 = np.delete(rms_time_2, index, axis = 0)
    
    rms_3_over = np.where(rms_3 >= np.mean(rms_3) + 0.2*np.mean(rms_3))
    rms_3 = np.delete(rms_3, rms_3_over, axis = 0)
    rms_time_3 = np.delete(rms_time, rms_3_over, axis = 0)
    index, peaks = scipy.signal.find_peaks(rms_3)
    rms_3 = np.delete(rms_3, index, axis = 0)
    rms_time_3 = np.delete(rms_time_3, index, axis = 0)
    
    times_sort = np.argsort(rms_time)
    rms_time = np.array(rms_time)[times_sort]
    bat_level = np.array(bat_level)[times_sort]
    
    times_sort_1 = np.argsort(rms_time_1)
    rms_time_1 = np.array(rms_time_1)[times_sort_1]
    rms_1 = np.array(rms_1)[times_sort_1]
    
    times_sort_2 = np.argsort(rms_time_2)
    rms_time_2 = np.array(rms_time_2)[times_sort_2]
    rms_2 = np.array(rms_2)[times_sort_2]

    times_sort_3 = np.argsort(rms_time_3)
    rms_time_3 = np.array(rms_time_3)[times_sort_3]
    rms_3 = np.array(rms_3)[times_sort_3]
    
    
    rms_times = []
    for t in rms_time:
        date = datetime.datetime.fromtimestamp(t).strftime("%m/%d/%Y, %H:%M:%S")[-8:]
        rms_times.append(datetime.datetime.strptime(date, "%H:%M:%S"))
       
    rms_times_1 = []
    for t in rms_time_1:
        date = datetime.datetime.fromtimestamp(t).strftime("%m/%d/%Y, %H:%M:%S")[-8:]
        rms_times_1.append(datetime.datetime.strptime(date, "%H:%M:%S"))
        
    rms_times_2 = []
    for t in rms_time_2:
        date = datetime.datetime.fromtimestamp(t).strftime("%m/%d/%Y, %H:%M:%S")[-8:]
        rms_times_2.append(datetime.datetime.strptime(date, "%H:%M:%S"))
        
    rms_times_3 = []
    for t in rms_time_3:
        date = datetime.datetime.fromtimestamp(t).strftime("%m/%d/%Y, %H:%M:%S")[-8:]
        rms_times_3.append(datetime.datetime.strptime(date, "%H:%M:%S"))
        
    
        
    return bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3

def average_over_timeinterval(bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3, delta=5):
    if not rms_times:
        return [], [], [], [], [], [], [], []

    start_dt = min(rms_times)
    end_dt = max(rms_times)
    delta = timedelta(minutes=delta)

    intervals = []
    while start_dt <= end_dt:
        intervals.append(start_dt)
        start_dt += delta

    intervals = np.array(intervals)
    rms_times = np.array(rms_times)
    rms_times_1 = np.array(rms_times_1)
    rms_times_2 = np.array(rms_times_2)
    rms_times_3 = np.array(rms_times_3)

    mean_bat_level = []
    mean_rms_1 = []
    mean_rms_2 = []
    mean_rms_3 = []

    for i in range(1, len(intervals)):
        points_rms = np.where(np.logical_and(rms_times >= intervals[i - 1], rms_times <= intervals[i]))
        mean_bat_level.append(np.mean(bat_level[points_rms]))

        points_rms_1 = np.where(np.logical_and(rms_times_1 >= intervals[i - 1], rms_times_1 <= intervals[i]))
        mean_rms_1.append(np.mean(rms_1[points_rms_1]))

        points_rms_2 = np.where(np.logical_and(rms_times_2 >= intervals[i - 1], rms_times_2 <= intervals[i]))
        mean_rms_2.append(np.mean(rms_2[points_rms_2]))

        points_rms_3 = np.where(np.logical_and(rms_times_3 >= intervals[i - 1], rms_times_3 <= intervals[i]))
        mean_rms_3.append(np.mean(rms_3[points_rms_3]))

    rms_time_1 = intervals[:-1]
    rms_time_2 = intervals[:-1]
    rms_time_3 = intervals[:-1]

    return mean_bat_level, intervals[:-1], mean_rms_1, mean_rms_2, mean_rms_3, rms_time_1, rms_time_2, rms_time_3

# def average_over_timeinterval(bat_level, rms_times, rms_1, rms_2, rms_3, rms_times_1, rms_times_2, rms_times_3, delta = 5):
#     '''
#     returns rms values averages over delta (10) minutes and the interval points
#     '''
    
    
#     if rms_times != []:
#         start_dt = min(rms_times)
#         end_dt = max(rms_times)
#         # difference between current and previous date
#         delta = timedelta(minutes = delta)

#         # store the dates between two dates in a list
#         intervals = []

#         while start_dt <= end_dt:
#             # add current date to list by converting  it to iso format
#             intervals.append(start_dt)
#             # increment start date by timedelta
#             start_dt += delta
         
#         mean_rms_1 = []
#         mean_rms_2 = []
#         mean_rms_3 = []
#         mean_bat_level = []
        
#         for i in range(len(intervals)):
#             points_rms = np.where(np.logical_and(np.array(rms_times)>=intervals[i-1], np.array(rms_times)<=intervals[i]))
#             mean_bat_level.append(np.mean(bat_level[points_rms]))
            
#             points_rms_1 = np.where(np.logical_and(np.array(rms_times_1)>=intervals[i-1], np.array(rms_times_1)<=intervals[i]))
#             mean_rms_1.append(np.mean(rms_1[points_rms_1]))
            
#             points_rms_2 = np.where(np.logical_and(np.array(rms_times_2)>=intervals[i-1], np.array(rms_times_2)<=intervals[i]))
#             mean_rms_2.append(np.mean(rms_2[points_rms_2]))
            
#             points_rms_3 = np.where(np.logical_and(np.array(rms_times_3)>=intervals[i-1], np.array(rms_times_3)<=intervals[i]))
#             mean_rms_3.append(np.mean(rms_3[points_rms_3]))
            
#         rms_time_1 = intervals
#         rms_time_2 = intervals
#         rms_time_3 = intervals
        
#         return mean_bat_level, intervals, mean_rms_1, mean_rms_2, mean_rms_3, rms_time_1, rms_time_2, rms_time_3

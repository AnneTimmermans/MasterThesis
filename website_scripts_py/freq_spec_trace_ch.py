import grand.dataio.root_trees as rt
import uproot
import awkward as ak
import numpy as np
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import ast
import scipy
from datetime import timedelta
from scipy.signal import find_peaks

def get_test_array(trawv):

    listevt = trawv.get_list_of_events()
    trawv.get_event(listevt[0][0], listevt[0][1])
    t_0 = trawv.trace_ch

    du_59 = None  # Initialize du_59
    for i, du_id in enumerate(trawv.du_id):
        if du_id == 59:
            du_59 = i
            break  # Found du_59, exit the loop

    if du_59 is not None:
        trace_0 = t_0[du_59][1] 
        
        sample_freq = 500  # [MHz]
        n_samples = len(trace_0)
        fft_freq = np.fft.rfftfreq(n_samples) * sample_freq  # [MHz]
        
        spectrum_0 = np.abs(np.fft.rfft(trace_0))
        peaks_0 = find_peaks(spectrum_0, distance=50, height=np.mean(spectrum_0) + 0.2 * np.mean(spectrum_0))[0]
        test_array = fft_freq[peaks_0[:4]]
        return test_array
    else:
        print("DU 59 not found, returning None")
        return []

def filter_weird_events(traces_np, du_number, test_array, GA_or_GP13, filter_transients = True):
    '''
    Input: 
            traces_np = np array([[trace_ch1],[trace_ch2],[trace_ch3]], ...)
    Output: 
            traces_new = np array([[trace_ch1],[trace_ch2],[trace_ch3]], ...)
                         filtered traces, where blocktraces and other weird traces are put to 0.
            
    '''
    if GA_or_GP13 == 'GA':
        channels = [0,1,2]

    if GA_or_GP13 == 'GP13':
        channels = [0,1,2]
    
    weirdos = []
    
    
    sample_freq = 500 # [MHz]
    n_samples = len(traces_np[0][0])
    fft_freq  = np.fft.rfftfreq(n_samples) * sample_freq # [MHz]

    for evt in range(0,len(traces_np)):
        for chs in range(3):
            ch = channels[chs]
            abs_0 = np.abs(traces_np[evt][ch])
            start_mean0 = np.mean(abs_0[0:len(abs_0)//6])
            end_mean0 = np.mean(abs_0[-len(abs_0)//6: -1])

            fft = np.fft.rfft(traces_np[evt][ch])
            psd = np.abs(fft)
            
            #filer chopped traces and early or late pulses
            if (np.abs(start_mean0 - end_mean0) >= start_mean0 or np.abs(start_mean0 - end_mean0) >= end_mean0):
                weirdos.append([du_number, ch,evt])
                continue

            #Block traces
            elif (fft_freq[np.where((psd == max(psd)).all())] == 250 or len(np.where(traces_np[evt][ch] == max(traces_np[evt][ch]))[0]) >= 200 or min(traces_np[evt][ch]) >= 0 or max(traces_np[evt][ch]) <= 0): 
                weirdos.append([du_number, ch,evt])
                continue
            
            # check for periodic behaviour
            peaks = find_peaks(psd, distance = 50, height = np.mean(psd) + 0.2 *np.mean(psd))[0]
            if len(peaks) >= 4 and len(test_array) >= 4:
                if np.allclose(test_array, fft_freq[peaks[:4]], rtol=0.1) or np.allclose(np.insert(test_array, 0, 1)[:4], fft_freq[peaks[:4]], rtol=0.1):
                    weirdos.append([du_number, ch,evt])
                    continue
            if len(peaks) >= 1:
                if np.allclose(fft_freq[343], fft_freq[peaks[-1]], rtol=0.01):
                    weirdos.append([du_number, ch,evt])
                    
            if len(peaks) >= 1:
                if 20 <=fft_freq[peaks[0]]<=35:#np.allclose(27.34375, fft_freq[peaks[0]], rtol=0.01):
                    weirdos.append([du_number, ch,evt])
                    
            if len(peaks) >= 2:
                if 20 <=fft_freq[peaks[0]]<=35:#np.allclose(27.34375, fft_freq[peaks[1]], rtol=0.01):
                    weirdos.append([du_number, ch,evt])
                
            #filter transient noise    
            if filter_transients:
                dlength = len(traces_np[evt][ch])
                dt= 1
                times=np.arange(dlength)*dt #ns
                indices = [i for i,v in enumerate(abs_0) if v > 5*np.std(traces_np[evt][ch])] # indices where amplitude > 5 x std
                t = times[indices]
                dt = [j-i for i, j in zip(t[:-1], t[1:])] 
                if (np.array(dt) <= 50).any and len(dt) >= 1:
                    weirdos.append([du_number, ch,evt])
    
    weirdos_cut = weirdos            
    
    if len(weirdos) != 0:            
        weirdos_cut = np.delete(weirdos, [0,1], axis = 1)
        traces_np = np.delete(traces_np, np.unique(weirdos_cut), axis = 0)
        
        
    return traces_np, weirdos_cut

def psd_freq(trace):
    sample_freq = 500 # [MHz]
    n_samples = len(trace)
    freq  = np.fft.rfftfreq(n_samples) * sample_freq # [MHz]

    gainlin = 10  # gainlin = VGA linear gain
    mfft = np.abs(np.fft.rfft(trace))#%10
    mfft = np.asarray(mfft/gainlin) # Back to voltage @ board input (gainlin = VGA linear gain)
    #
    # Now to power
    pfft = np.zeros(np.shape(mfft)) # Back to 1MHz step
    ib = len(mfft)
    pfft = mfft**2/(ib**2)
    dnu = (freq[1]-freq[0])/1 # MHz/MHz unitless
    pfft = pfft/dnu  

    return pfft

def avg_freq_trace(file, GA_or_GP13, Filter=False):
    '''
    Input: 
            file = file name of the root file
            filter = boolean to determine whether the events are filtered or not
            
    Output: 
            FREQ = {DU: [[avg_freq_x],[avg_freq_y],[avg_freq_z]]} 
            TRACE = {DU: [[avg_trace_x],[avg_trace_y],[avg_trace_z]]} 
            
    '''
    if GA_or_GP13 == 'GA':
        ch1 = 0
        ch2 = 1
        ch3 = 2

    if GA_or_GP13 == 'GP13':
        ch1 = 1
        ch2 = 2
        ch3 = 3

    FREQ = {}
    TRACE = {}
    try:
        TADC  = rt.TADC(file)
        df = rt.DataFile(file)
        TRAWV = df.trawvoltage
        trawv = uproot.open(file)['trawvoltage']   # Extract the trawvoltage tree 
        tadc =  uproot.open(file)['tadc']
        
        test_array = get_test_array(TRAWV)
        # test_array = get_test_array(TADC)

        duid = trawv["du_id"].array() 
        du_list = np.unique(ak.flatten(duid))
        du_list = np.trim_zeros(du_list)

        traces_array_trawv = trawv["trace_ch"].array()  # get the traces array
        traces_array =  tadc["trace_ch"].array()
        trigger_times = trawv["du_seconds"].array()  # get the time array



        for du_number in du_list:
            
            idx_du = duid == du_number  # creates an boolean ak array to know if/where the du_number is in the events
            idx_dupresent = ak.where(ak.sum(idx_du, axis=1)) # this computes whether the du is present in the event

            traces_array_du = traces_array[idx_du]  # gets the traces of the correct du
            trigger_time_du = trigger_times[idx_du]  # gets the triggertime of the correct du

            result = traces_array_du[idx_dupresent] # removes the events where the DU is not present
            trigger_time = trigger_time_du[idx_dupresent]

            traces_np = result[:, 0, ch1:ch3+1].to_numpy() # now results should be result and can be "numpied"
            trigger_time = trigger_time[:, 0].to_numpy()
            
            
            # if Filter:
            #     traces_np, weirdos_cut = filter_weird_events(traces_np, du_number, test_array, GA_or_GP13)
            
            
            Freq_x = []
            Freq_y = []
            Freq_z = []
            
            Trace_x = []
            Trace_y = []
            Trace_z = []
            
            for evt in range(0,len(traces_np)):
                        
                Freq_x.append(psd_freq(traces_np[evt][0]*(0.9/8192)))#np.abs(np.fft.rfft(traces_np[evt][0])))
                Freq_y.append(psd_freq(traces_np[evt][1]*(0.9/8192)))#np.abs(np.fft.rfft(traces_np[evt][1])))
                Freq_z.append(psd_freq(traces_np[evt][2]*(0.9/8192)))#np.abs(np.fft.rfft(traces_np[evt][2])))
                
                Trace_x.append(traces_np[evt][0])
                Trace_y.append(traces_np[evt][1])
                Trace_z.append(traces_np[evt][2])
                

            DU_freq = [np.mean(np.asarray(Freq_x),axis=0), np.mean(np.asarray(Freq_y),axis=0), np.mean(np.asarray(Freq_z),axis=0)]
            DU_trace = [np.mean(np.asarray(Trace_x),axis=0), np.mean(np.asarray(Trace_y),axis=0),np.mean(np.asarray(Trace_z),axis=0)]
                
            FREQ["DU_{}".format(du_number)] = [len(traces_np),DU_freq]
            TRACE["DU_{}".format(du_number)] = [len(traces_np),DU_trace]
    
    except:
        FREQ = {}
        TRACE = {}
              
    return FREQ, TRACE  # , fft_freq
   
def freq_to_npz(file, GA_or_GP13, Filter=False):
    '''
    Input: 
            file = file name of the root file
            filter = boolean to determine whether the events are filtered or not
            
    Output: 
            FREQ = {DU: [[avg_freq_x],[avg_freq_y],[avg_freq_z]]} 
            TRACE = {DU: [[avg_trace_x],[avg_trace_y],[avg_trace_z]]} 
            
    '''
    if (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5])) == True) and (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5])) == True):
        TRACE = np.load('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5]), allow_pickle=True)
        FREQ = np.load('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5]), allow_pickle=True)
        print('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz and /pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz exist'.format(file[-19:-5], file[-19:-5]))

    if (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5])) == False):
        FREQ, TRACE, fft_freq = avg_freq_trace(file, GA_or_GP13, Filter=False)
        np.savez('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5]), ** TRACE)
        np.savez('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5]), ** FREQ)

    elif (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5])) == False):
        FREQ, TRACE, fft_freq = avg_freq_trace(file, GA_or_GP13, Filter=False)
        np.savez('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5]), ** FREQ)
        np.savez('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5]), ** TRACE)
        
    return FREQ, TRACE

def weighted_average(freq_data):
    total_weight = 0
    weighted_sum = np.zeros_like(freq_data[0][1])
    
    for weight, freq_list in freq_data:
        if not np.shape(freq_list) == (3,):
            total_weight += weight
            weighted_sum += weight * np.array(freq_list)

    if total_weight != 0:
        weighted_avg = weighted_sum / total_weight
        return [total_weight, weighted_avg.tolist()]
    else:
        return None  # Handle the case where total_weight is zero (avoid division by zero)

# def avg_freq_trace_day_night(file, GA_or_GP13, day_start_hour, day_end_hour, Filter=False):
#     '''
#     Input: 
#             file = file name of the root file
#             GA_or_GP13 = 'GA' or 'GP13'
#             Filter = boolean to determine whether the events are filtered or not
#             day_start_hour = start hour of the day (default is 6 AM)
#             day_end_hour = end hour of the day (default is 6 PM)
            
#     Output: 
#             FREQ = {DU: [[avg_freq_x],[avg_freq_y],[avg_freq_z]]} 
#             TRACE = {DU: [[avg_trace_x],[avg_trace_y],[avg_trace_z]]} 
#     '''
#     if GA_or_GP13 == 'GA':
#         ch1 = 0
#         ch2 = 1
#         ch3 = 2

#     if GA_or_GP13 == 'GP13':
#         ch1 = 1
#         ch2 = 2
#         ch3 = 3

#     FREQ = {}
#     TRACE = {}
    
#     try:
#         TADC = rt.TADC(file)
#         df = rt.DataFile(file)
#         TRAWV = df.trawvoltage
#         trawv = uproot.open(file)['trawvoltage']   # Extract the trawvoltage tree 
#         tadc =  uproot.open(file)['tadc']

#         duid = trawv["du_id"].array() 
#         du_list = np.unique(ak.flatten(duid))
#         du_list = np.trim_zeros(du_list)

#         traces_array_trawv = trawv["trace_ch"].array()  # get the traces array
#         traces_array =  tadc["trace_ch"].array()
#         trigger_times = trawv["du_seconds"].array()  # get the time array

#         for du_number in du_list:
#             idx_du = duid == du_number
#             idx_dupresent = ak.where(ak.sum(idx_du, axis=1))

#             traces_array_du = traces_array[idx_du]
#             trigger_time_du = trigger_times[idx_du]
            
#             # Filter events based on day time
#             day_time_mask = (trigger_time_du[:, 0] >= day_start_hour * 3600) & (trigger_time_du[:, 0] <= day_end_hour * 3600)
#             print(day_start_hour,len(day_time_mask))
#             traces_array_du = traces_array_du[day_time_mask]
#             trigger_time_du = trigger_time_du[day_time_mask]
#             print("Number of events after day time filtering:", len(traces_array_du))
#             print("Trigger times after day time filtering:", trigger_time_du)

#             result = traces_array_du[idx_dupresent]
#             trigger_time = trigger_time_du[idx_dupresent]

#             traces_np = result[:, 0, ch1:ch3+1].to_numpy()
#             trigger_time = trigger_time[:, 0].to_numpy()
            
#             Freq_x = [psd_freq(evt[0]*(0.9/8192)) for evt in traces_np]
#             Freq_y = [psd_freq(evt[1]*(0.9/8192)) for evt in traces_np]
#             Freq_z = [psd_freq(evt[2]*(0.9/8192)) for evt in traces_np]

#             Trace_x = [evt[0] for evt in traces_np]
#             Trace_y = [evt[1] for evt in traces_np]
#             Trace_z = [evt[2] for evt in traces_np]

#             DU_freq = [np.mean(Freq_x, axis=0), np.mean(Freq_y, axis=0), np.mean(Freq_z, axis=0)]
#             DU_trace = [np.mean(Trace_x, axis=0), np.mean(Trace_y, axis=0), np.mean(Trace_z, axis=0)]

#             FREQ["DU_{}".format(du_number)] = [len(traces_np), DU_freq]
#             TRACE["DU_{}".format(du_number)] = [len(traces_np), DU_trace]

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         FREQ = {}
#         TRACE = {}

              
#     return FREQ, TRACE, len(traces_array_du)

import datetime

# def avg_freq_trace_day_night(file, GA_or_GP13,day_night, Filter=False):
#     '''
#     Input: 
#         file = file name of the root file
#         GA_or_GP13 = 'GA' or 'GP13'
#         Filter = boolean to determine whether the events are filtered or not
#         day_start_hour = start hour of the day (default is 6 AM)
#         day_end_hour = end hour of the day (default is 6 PM)
        
#     Output: 
#         FREQ = {DU: [[avg_freq_x],[avg_freq_y],[avg_freq_z]]} 
#         TRACE = {DU: [[avg_trace_x],[avg_trace_y],[avg_trace_z]]} 
#     '''
#     if GA_or_GP13 == 'GA':
#         ch1 = 0
#         ch2 = 1
#         ch3 = 2
#         day_start_hour = 8+3 #utc + 3
#         day_end_hour = 20+3 # utc + 3

#     if GA_or_GP13 == 'GP13':
#         ch1 = 1
#         ch2 = 2
#         ch3 = 3
#         day_start_hour = 00 # utc - 8
#         day_end_hour = 12# utc - 8

#     FREQ = {}
#     TRACE = {}
    
#     if True:#try:
#         TADC = rt.TADC(file)
#         df = rt.DataFile(file)
#         TRAWV = df.trawvoltage
#         trawv = uproot.open(file)['trawvoltage']   # Extract the trawvoltage tree 
#         tadc =  uproot.open(file)['tadc']

#         duid = trawv["du_id"].array() 
#         du_list = np.unique(ak.flatten(duid))
#         du_list = np.trim_zeros(du_list)

#         traces_array_trawv = trawv["trace_ch"].array()  # get the traces array
#         traces_array =  tadc["trace_ch"].array()
#         trigger_times = trawv["du_seconds"].array()  # get the time array

#         for du_number in du_list:
#             try:
#                 idx_du = duid == du_number
#                 idx_dupresent = ak.where(ak.sum(idx_du, axis=1))
                
#                 print("A",np.shape(trigger_times[:,0]))
#                 traces_array_du = traces_array[idx_du]
#                 trigger_time_du = trigger_times[:,0][idx_du]
#                 print("B",np.shape(trigger_time_du))

#                 # Filter events based on day time
#                 day_start_time = datetime.datetime.strptime(f"{day_start_hour:02d}:00", "%H:%M").time()
#                 day_end_time = datetime.datetime.strptime(f"{day_end_hour:02d}:00", "%H:%M").time()

#                 day_time_mask = []
#                 for time in trigger_time_du[:, 0]:
#                     dt_time = datetime.datetime.utcfromtimestamp(time).time()
#                     if  day_start_time <= dt_time <= day_end_time: #day_start_time <= dt_time or dt_time <= day_end_time:# or
#                         if day_night == 'day':
#                             day_time_mask.append(True)
#                         if day_night == 'night':
#                             day_time_mask.append(False)

#                         # print(f"{dt_time} WEL binnen {day_start_hour} en {day_end_hour}")
#                     else:
#                         # print(f"{dt_time} niet binnen {day_start_hour} en {day_end_hour}")
#                         if day_night == 'day':
#                             day_time_mask.append(False)
#                         if day_night == 'night':
#                             day_time_mask.append(True)

#                 print(f"false daytime:{len(np.where(day_time_mask == False))}\n total events {len(trigger_time_du)}")
#                 print(f"min time {min(trigger_time_du[:, 0])} max time {max(trigger_time_du[:, 0])}")
#                 traces_array_du = traces_array_du[day_time_mask]
#                 trigger_time_du = trigger_time_du[day_time_mask]
#                 idx_du = idx_du[day_time_mask]
#                 idx_dupresent = ak.where(ak.sum(idx_du, axis=1))

#                 print(f"Number of events during {day_night}:", len(traces_array_du))
#                 print("Trigger times after day time filtering:", trigger_time_du)

#                 result = traces_array_du[idx_dupresent]
#                 # trigger_time = trigger_time_du[idx_dupresent]

#                 traces_np = result[:, 0, ch1:ch3+1].to_numpy()
#                 # trigger_time = trigger_time[:, 0].to_numpy()

#                 Freq_x = [psd_freq(evt[0]*(0.9/8192)) for evt in traces_np]
#                 Freq_y = [psd_freq(evt[1]*(0.9/8192)) for evt in traces_np]
#                 Freq_z = [psd_freq(evt[2]*(0.9/8192)) for evt in traces_np]

#                 Trace_x = [evt[0] for evt in traces_np]
#                 Trace_y = [evt[1] for evt in traces_np]
#                 Trace_z = [evt[2] for evt in traces_np]

#                 DU_freq = [np.mean(Freq_x, axis=0), np.mean(Freq_y, axis=0), np.mean(Freq_z, axis=0)]
#                 DU_trace = [np.mean(Trace_x, axis=0), np.mean(Trace_y, axis=0), np.mean(Trace_z, axis=0)]

#                 FREQ["DU_{}".format(du_number)] = [len(traces_np), DU_freq]
#                 TRACE["DU_{}".format(du_number)] = [len(traces_np), DU_trace]

#             except Exception as e:
#                 print(f"An error occurred: {e}")

#     return FREQ, TRACE
import datetime
import datetime

def avg_freq_trace_day(file, GA_or_GP13, Filter=False):
    '''
    Input: 
        file = file name of the root file
        filter = boolean to determine whether the events are filtered or not
            
    Output: 
        FREQ = {DU: [[avg_freq_x],[avg_freq_y],[avg_freq_z]]} 
        TRACE = {DU: [[avg_trace_x],[avg_trace_y],[avg_trace_z]]} 
    '''
    if GA_or_GP13 == 'GA':
        ch1 = 0
        ch2 = 1
        ch3 = 2
        start = 6+3 # utc + 3 
        end = 18 +3 # utc + 3

    if GA_or_GP13 == 'GP13':
        ch1 = 1
        ch2 = 2
        ch3 = 3
        start = 22 # utc -8
        end = 18 -8 # utc -8

    FREQ = {}
    TRACE = {}
    try:
        TADC = rt.TADC(file)
        df = rt.DataFile(file)
        TRAWV = df.trawvoltage
        trawv = uproot.open(file)['trawvoltage']   # Extract the trawvoltage tree 
        tadc = uproot.open(file)['tadc']
        
        test_array = get_test_array(TRAWV)

        duid = trawv["du_id"].array() 
        du_list = np.unique(ak.flatten(duid))
        du_list = np.trim_zeros(du_list)

        traces_array_trawv = trawv["trace_ch"].array()  # get the traces array
        traces_array = tadc["trace_ch"].array()
        trigger_times = trawv["du_seconds"].array()  # get the time array

        for du_number in du_list:
            idx_du = duid == du_number  # creates a boolean ak array to know if/where the du_number is in the events
            idx_dupresent = ak.where(ak.sum(idx_du, axis=1)) # this computes whether the du is present in the event

            traces_array_du = traces_array[idx_du]  # gets the traces of the correct du
            trigger_time_du = trigger_times[idx_du]  # gets the trigger time of the correct du

            result = traces_array_du[idx_dupresent] # removes the events where the DU is not present
            trigger_time = trigger_time_du[idx_dupresent]

            traces_np = result[:, 0, ch1:ch3+1].to_numpy() # now results should be result and can be "numpied"
            # trigger_time = trigger_time[:, 0].to_numpy()
            # Assuming 'start' and 'end' are the start and end times in 24-hour format
            start_time = datetime.strptime(f"{start}:00", "%H:%M")
            end_time = datetime.strptime(f"{end}:00", "%H:%M")

            # ...

            # Convert trigger times to datetime objects
            trigger_datetime = [datetime.utcfromtimestamp(ts) for ts in trigger_time[:, 0].to_numpy()]

            # Filter events based on datetime (daytime: 6:00 to 18:00)
            day_time_mask = [start_time <= dt.time() < end_time for dt in trigger_datetime]
            trigger_time = trigger_time[day_time_mask]
            traces_np = traces_np[day_time_mask]
            # trigger_time = trigger_time[day_time_mask]

            Freq_x = []
            Freq_y = []
            Freq_z = []

            Trace_x = []
            Trace_y = []
            Trace_z = []

            for evt in range(len(traces_np)):
                Freq_x.append(psd_freq(traces_np[evt][0] * (0.9/8192)))
                Freq_y.append(psd_freq(traces_np[evt][1] * (0.9/8192)))
                Freq_z.append(psd_freq(traces_np[evt][2] * (0.9/8192)))

                Trace_x.append(traces_np[evt][0])
                Trace_y.append(traces_np[evt][1])
                Trace_z.append(traces_np[evt][2])

            DU_freq = [np.mean(np.asarray(Freq_x), axis=0), np.mean(np.asarray(Freq_y), axis=0), np.mean(np.asarray(Freq_z), axis=0)]
            DU_trace = [np.mean(np.asarray(Trace_x), axis=0), np.mean(np.asarray(Trace_y), axis=0), np.mean(np.asarray(Trace_z), axis=0)]

            FREQ["DU_{}".format(du_number)] = [len(traces_np), DU_freq]
            TRACE["DU_{}".format(du_number)] = [len(traces_np), DU_trace]

    except:
        FREQ = {}
        TRACE = {}
              
    return FREQ, TRACE

def avg_freq_trace_night(file, GA_or_GP13, Filter=False):
    '''
    Input: 
        file = file name of the root file
        filter = boolean to determine whether the events are filtered or not
            
    Output: 
        FREQ = {DU: [[avg_freq_x],[avg_freq_y],[avg_freq_z]]} 
        TRACE = {DU: [[avg_trace_x],[avg_trace_y],[avg_trace_z]]} 
    '''
    if GA_or_GP13 == 'GA':
        ch1 = 0
        ch2 = 1
        ch3 = 2
        start = 20 + 3 # utc -8
        end = 8 + 3 # utc -8

    if GA_or_GP13 == 'GP13':
        ch1 = 1
        ch2 = 2
        ch3 = 3
        start = 20 - 8 # utc -8
        end = 00 # utc -8

    FREQ = {}
    TRACE = {}
    try:
        TADC = rt.TADC(file)
        df = rt.DataFile(file)
        TRAWV = df.trawvoltage
        trawv = uproot.open(file)['trawvoltage']   # Extract the trawvoltage tree 
        tadc = uproot.open(file)['tadc']
        
        test_array = get_test_array(TRAWV)

        duid = trawv["du_id"].array() 
        du_list = np.unique(ak.flatten(duid))
        du_list = np.trim_zeros(du_list)

        traces_array_trawv = trawv["trace_ch"].array()  # get the traces array
        traces_array = tadc["trace_ch"].array()
        trigger_times = trawv["du_seconds"].array()  # get the time array

        for du_number in du_list:
            idx_du = duid == du_number  # creates a boolean ak array to know if/where the du_number is in the events
            idx_dupresent = ak.where(ak.sum(idx_du, axis=1)) # this computes whether the du is present in the event

            traces_array_du = traces_array[idx_du]  # gets the traces of the correct du
            trigger_time_du = trigger_times[idx_du]  # gets the trigger time of the correct du

            result = traces_array_du[idx_dupresent] # removes the events where the DU is not present
            trigger_time = trigger_time_du[idx_dupresent]

            traces_np = result[:, 0, ch1:ch3+1].to_numpy() # now results should be result and can be "numpied"
            trigger_time = trigger_time[:, 0].to_numpy()

            # Filter events based on trigger times (night time: 20:00 to 8:00)
            night_time_mask = np.logical_or(trigger_time >= start * 3600, trigger_time < end * 3600)
            traces_np = traces_np[night_time_mask]
            trigger_time = trigger_time[night_time_mask]

            Freq_x = []
            Freq_y = []
            Freq_z = []

            Trace_x = []
            Trace_y = []
            Trace_z = []

            for evt in range(len(traces_np)):
                Freq_x.append(psd_freq(traces_np[evt][0] * (0.9/8192)))
                Freq_y.append(psd_freq(traces_np[evt][1] * (0.9/8192)))
                Freq_z.append(psd_freq(traces_np[evt][2] * (0.9/8192)))

                Trace_x.append(traces_np[evt][0])
                Trace_y.append(traces_np[evt][1])
                Trace_z.append(traces_np[evt][2])

            DU_freq = [np.mean(np.asarray(Freq_x), axis=0), np.mean(np.asarray(Freq_y), axis=0), np.mean(np.asarray(Freq_z), axis=0)]
            DU_trace = [np.mean(np.asarray(Trace_x), axis=0), np.mean(np.asarray(Trace_y), axis=0), np.mean(np.asarray(Trace_z), axis=0)]

            FREQ["DU_{}".format(du_number)] = [len(traces_np), DU_freq]
            TRACE["DU_{}".format(du_number)] = [len(traces_np), DU_trace]

    except:
        FREQ = {}
        TRACE = {}
              
    return FREQ, TRACE

    '''
    Input: 
        file = file name of the root file
        filter = boolean to determine whether the events are filtered or not
            
    Output: 
        FREQ = {DU: [[avg_freq_x],[avg_freq_y],[avg_freq_z]]} 
        TRACE = {DU: [[avg_trace_x],[avg_trace_y],[avg_trace_z]]} 
    '''
    if GA_or_GP13 == 'GA':
        ch1 = 0
        ch2 = 1
        ch3 = 2
        B = 20 +3 # utc +3
        A = 8 +3 # utc +3
        if day_night == 'night':
            A = 20 +3 # utc +3
            B = 8 +3 # utc +3
        

    if GA_or_GP13 == 'GP13':
        ch1 = 1
        ch2 = 2
        ch3 = 3
        B = 20 -8 # utc -8
        A = 00 # utc -8
        if day_night == 'night':
            A = 20 -8 # utc -8
            B = 00 # utc -8

    FREQ = {}
    TRACE = {}
    try:
        TADC = rt.TADC(file)
        df = rt.DataFile(file)
        TRAWV = df.trawvoltage
        trawv = uproot.open(file)['trawvoltage']   # Extract the trawvoltage tree 
        tadc = uproot.open(file)['tadc']
        
        test_array = get_test_array(TRAWV)

        duid = trawv["du_id"].array() 
        du_list = np.unique(ak.flatten(duid))
        du_list = np.trim_zeros(du_list)

        traces_array_trawv = trawv["trace_ch"].array()  # get the traces array
        traces_array = tadc["trace_ch"].array()
        trigger_times = trawv["du_seconds"].array()  # get the time array

        for du_number in du_list:
            idx_du = duid == du_number  # creates a boolean ak array to know if/where the du_number is in the events
            idx_dupresent = ak.where(ak.sum(idx_du, axis=1)) # this computes whether the du is present in the event

            traces_array_du = traces_array[idx_du]  # gets the traces of the correct du
            trigger_time_du = trigger_times[idx_du]  # gets the trigger time of the correct du

            result = traces_array_du[idx_dupresent] # removes the events where the DU is not present
            trigger_time = trigger_time_du[idx_dupresent]

            traces_np = result[:, 0, ch1:ch3+1].to_numpy() # now results should be result and can be "numpied"
            trigger_time = trigger_time[:, 0].to_numpy()

            # Filter events based on trigger times (night time: 20:00 to 8:00)
            night_time_mask = np.logical_or(trigger_time >= A * 3600, trigger_time < B * 3600)
            traces_np = traces_np[night_time_mask]
            trigger_time = trigger_time[night_time_mask]

            Freq_x = []
            Freq_y = []
            Freq_z = []

            Trace_x = []
            Trace_y = []
            Trace_z = []

            for evt in range(len(traces_np)):
                Freq_x.append(psd_freq(traces_np[evt][0] * (0.9/8192)))
                Freq_y.append(psd_freq(traces_np[evt][1] * (0.9/8192)))
                Freq_z.append(psd_freq(traces_np[evt][2] * (0.9/8192)))

                Trace_x.append(traces_np[evt][0])
                Trace_y.append(traces_np[evt][1])
                Trace_z.append(traces_np[evt][2])

            DU_freq = [np.mean(np.asarray(Freq_x), axis=0), np.mean(np.asarray(Freq_y), axis=0), np.mean(np.asarray(Freq_z), axis=0)]
            DU_trace = [np.mean(np.asarray(Trace_x), axis=0), np.mean(np.asarray(Trace_y), axis=0), np.mean(np.asarray(Trace_z), axis=0)]

            FREQ["DU_{}".format(du_number)] = [len(traces_np), DU_freq]
            TRACE["DU_{}".format(du_number)] = [len(traces_np), DU_trace]

    except:
        FREQ = {}
        TRACE = {}
              
    return FREQ, TRACE
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

    gainlin = 20  # gainlin = VGA linear gain
    mfft = np.abs(np.fft.rfft(trace))%10
    mfft = np.asarray(mfft/gainlin) # Back to voltage @ board input (gainlin = VGA linear gain)
    #
    # Now to power
    pfft = np.zeros(np.shape(mfft)) # Back to 1MHz step
    ib = len(mfft)
    pfft = mfft**2/(ib**2)
    dnu = (freq[1]-freq[0])/1 # MHz/MHz unitless
    pfft = pfft/dnu  

    return pfft

def avg_freq_trace(file, GA_or_GP13, Filter=True):
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
            
            
            if Filter:
                traces_np, weirdos_cut = filter_weird_events(traces_np, du_number, test_array, GA_or_GP13)
            
            
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
                
            FREQ["DU_{}".format(du_number)] = DU_freq
            TRACE["DU_{}".format(du_number)] = DU_trace
    
    except:
        FREQ = {}
        TRACE = {}
    # # Compute the FFT normalization
    # sample_freq = 500 # [MHz]
    # for trace in traces_np:
    #     if trace:
    #         # If the trace is not empty, set n_samples to its length and break the loop
    #         n_samples = len(trace[0])
    #         break
    # fft_freq  = np.fft.rfftfreq(n_samples) * sample_freq # [MHz]
              
    return FREQ, TRACE  # , fft_freq
    
def freq_to_npz(file, GA_or_GP13, Filter=True):
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
        FREQ, TRACE, fft_freq = avg_freq_trace(file, GA_or_GP13, Filter=True)
        np.savez('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5]), ** TRACE)
        np.savez('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5]), ** FREQ)

    elif (os.path.exists('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5])) == False):
        FREQ, TRACE, fft_freq = avg_freq_trace(file, GA_or_GP13, Filter=True)
        np.savez('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_freq.npz'.format(file[-19:-5]), ** FREQ)
        np.savez('/pbs/home/a/atimmerm/GRAND/monitoring_website/website_scripts_py/avg_freq_trace_data/{}_avg_trace.npz'.format(file[-19:-5]), ** TRACE)
        
    return FREQ, TRACE

import numpy as np
from scipy.stats import norm
from scipy.ndimage import median_filter
import scipy.io as spio
import tensorflow as tf

def bandpass_filter_fft(signal, fs, lowcut, highcut):
    """
    Applies a bandpass filter to a signal using FFT and cut off frequencies.

    Inputs:
    - signal: list of floats, original time domain signal
    - fs: integer, the sampling frequency of the signal
    - lowcut: integer, the lower frequency cutoff for the filter
    - highcut: integer, the upper frequency cutoff for the filter

    Outputs:
    - iftt: 1D numpy array, filtered signal
    """
    # Calculate FFT of the signal and frequencies relative to sampling rate
    fft = np.fft.fft(signal)
    x_freq = np.fft.fftfreq(len(signal), 1/fs)

    # Shift the zero frequency component to the center
    fft_shifted = np.fft.fftshift(fft)
    freq_shifted = np.fft.fftshift(x_freq)

    # Remove components of FFT outside frequency cut-offs. 
    mask = (freq_shifted >= lowcut) & (freq_shifted <= highcut)
    signal_fft_filtered_shifted = fft_shifted * mask

    #Perform inverse FFT to convert back to time domain signal. 
    iftt = np.fft.ifft(np.fft.ifftshift(signal_fft_filtered_shifted)).real
    return iftt

def filter_data(domain):

    """
    Filters 1D signal using a bandpass and median filter,
    Inputs:
    - domain: list of floats, original signal

    Outputs:
    - norm_filt: list of floats, normalised filtered signal
    """

    # Sets the bandpass cut off frequencies and median filter window size depending on the 
    # dataset to be tested. 

    #D2
    (median_filter_window, low_cut_off, high_cut_off) = (7, 1, 2900)
    #D3
    #(median_filter_window, low_cut_off, high_cut_off) = (11, 10, 2600)
    #D4
    #(median_filter_window, low_cut_off, high_cut_off) = (15, 10, 2900)
    #D5
    #(median_filter_window, low_cut_off, high_cut_off) = (19, 15, 2650)
    #D6
    #(median_filter_window, low_cut_off, high_cut_off) = (27, 40, 2400)

    # Apply bandpass filter and median filter to the signal to remove high frequency noise.
    freq_filtered = bandpass_filter_fft(domain, 25000, low_cut_off, high_cut_off)
    med_filtered = median_filter(freq_filtered, size=median_filter_window, mode='reflect')

    # Normalise signal to set peak values between 0-1. 
    normalise = max(med_filtered)
    norm_filt = [val/normalise for val in med_filtered]

    return norm_filt

def get_peaks(domain):
    """
    Finds the indices of the peaks and there beginnings in a signal.

    Inputs:
    - domain: list of floats, data signal to be analysed

    Outputs:
    - peak_start: list of integers, indices of start of each peak
    - max_points: list of integers, indices of maximum value in each peak
    """

    # Sort the original signal by amplitude, ensuring the point number for each is arranged in the same order.  
    x = np.arange(1, len(domain))
    combined_lists = list(zip(domain, x))
    sorted_lists = sorted(combined_lists, key=lambda br: br[0]) 
    sorted_d, sorted_ind = zip(*sorted_lists)        

    # Calculates average value, mu, and standard devation of numerical values in the signal. 
    mu, std = norm.fit(sorted_d)

    # Find the index of the first value that is higher than the mean + a constant * standard deviation
    # depending on the dataset. The higher values are more likely to belong to peaks. 

    #D2 
    const = 0.5
    #D3 
    #const = 1
    #D4 
    #const = 1.2
    #D5 
    #const = 1.8
    #D6 
    #const = 1.6
    
    high_peak_cutoff = round(mu+const*std, 4)

    # Find the index of the first value that is lower than the high peak cutoff based on the sorted lists.
    rev_d = reversed(sorted_d)
    for idx, val in enumerate(rev_d):     
        if round(val,4) == high_peak_cutoff:
            higher_cutoff_index = len(sorted_d) - idx 
            break

    # Takes the indexes of the highest amplitudes in the signal from the cutoff index. 
    # Sorts them in back in order of index.  
    tocheck_indexes = sorted(sorted_ind[higher_cutoff_index: len(sorted_d)-1])
    print("Number of peaks to check: ", len(tocheck_indexes))

    # The list of indexes can contain multiple values for the same peak. 
    # They must be grouped together with the beginning and maximum values extracted

    # Initiate list to store peak starts and maximum values.
    peak_start = []
    max_points = []
    # Initiates a temporary holding list for values belonging to the same peak.
    temp = [tocheck_indexes[0]]
    prev_val = tocheck_indexes[0]

    # Intiial state of signal is rising, R. 
    state = "R" 
    # Keeps track of how many values are falling consecutively.
    consec_falling = 0 

    #Iterates through each index value
    for count, index in enumerate(tocheck_indexes, start=1):
        #If indexes are close together, they can be assumed to be part of the same peak. 
        if (index - prev_val)  < 3  and count!= len(domain):
            # Compares with the previous value if the signla is rising. 
            if (domain[index] > domain[prev_val]):
                # If rising, breaks the string of falling values. 
                consec_falling = 0
                # If the state has gone from falling to rising, a minimum has been discovered and indicates 
                # the start of a peak. 
                if state == "F":   
                    # Calculates the index of the maximum value and checks the results aren't flat
                    # or very narrow. 
                    if (len(temp) > 2) & (min(temp)!= max(temp)):
                        drange = [domain[t_p] for t_p in temp]
                        point = temp[drange.index(max(drange))]
                    else:
                        point = temp[0]
                    # Store the maximum peak index and the start position. 
                    max_points.append(point)
                    peak_start.append(get_peak_start(domain, point))
                    # Temporary peak list restarted from the current index value and state returned back to 
                    # increasing to restart the process. 
                    temp = []
                    state = "R"
                temp.append(index)

            # If the signal is falling and the temporary peak is big enough (not outlier).
            elif (domain[index] < domain[prev_val]) & (len(temp)>2):
                # Increments the consecutive falling value counter. Only changes state after 2 values
                # to ensure noise doesn't interrupt falling.
                consec_falling = consec_falling + 1
                if (consec_falling >= 2):
                    state = "F"
                temp.append(index)
        
        #If the two indexes are not part of the same peak:
        else:
            # Resets state to rising and extracts peak start and maximum as before. 
            state = "R"
            if (len(temp) > 1) & (min(temp)!= max(temp)):
                drange = [domain[t_p] for t_p in temp]
                point = temp[drange.index(max(drange))]
            else:
                point = temp[0]
            max_points.append(point)
            peak_start.append(get_peak_start(domain, point))
            temp = [index]
        prev_val = index
 
    print("Number of peaks found: ", len(peak_start))
    return (peak_start, max_points)

def get_peak_start(domain, peak):
    """
    Finds the start index of a peak given it's index in a signal. 

    Inputs:
    - domain: list of floats, signal containing the peak
    - peak: integer, index of the peak in the signal

    Outputs:
    - start_index: integer, index of the start of the peak
    """

    # Initialize counter from peak position and track consecutive rising values.
    peak_start_index = peak
    consec_rising = 0

    # Iterate backwards within the allowed peak .
    for _ in range(50):
        val = domain[peak_start_index]
        prev_val = domain[peak_start_index - 1]
        # Check if the signal is rising by comparing current and previous amplitude
        if prev_val > val:
            consec_rising += 1
            # If rising for at least two consecutive values, take the value before the rise (local trough)
            # and break. 
            if consec_rising >= 2:
                peak_start_index = peak_start_index + 3
                break
        # Reset the consecutive rising counter if the signal is not rising
        else:
            consec_rising = 0
        # Move to the previous index
        peak_start_index -= 1
    
    return peak_start_index

#------------------MAIN FUNCTION--------------------

# Load the target dataset and extract the time domain signal.
mat = spio.loadmat('D2.mat', squeeze_me=True)
domain = mat['d']       #Raw time domain recording, sample rate 25kHz

# Filter the signal to remove noise and normalise. 
d = filter_data(domain)

# Find the indexes of the starts and maximums of peaks 
(peak_start, max_points) = get_peaks(d)

# Initialise the target Index and Class lists. 
Indexes = []
Classes = []

# Load the CNN model from local folder. 
model = tf.keras.models.load_model("CNN_Model.h5")

# Initiate spike window width and number of values before/after peak, matches input expected by CNN.
window_width = 100
window_before = 40
window_finish = window_width-window_before

# Iterates through all peak values location from peak detection function
for count, peak_max_index in enumerate(max_points):
    if peak_max_index > window_before:

        # Obtains window surrounding the peak value. Normalises so the surrouding signal noise 
        # is about y = 0. 
        window = d[peak_max_index-window_before: peak_max_index + window_finish]
        Start_DC_Comp = np.mean(window[0:int(window_before*0.6)])
        window_norm = [a-Start_DC_Comp for a in window]
        check = np.array(window_norm)
        check = check.reshape(1,window_width, 1) 

        # Obtains probabilities of a window being a class. Selects the one with maximum probability.
        predictions_one_hot = list(model.predict(check, verbose=0)[0])
        prediction = predictions_one_hot.index(max(predictions_one_hot))


        # Previous submissions showed that some classes can be picked more than others, mostly 3 and 4. 
        # Therefore any unconfident instances of classes 0, 3 or 4 are replaced with the second most
        # likely class for an index. 
        # This is repeated for classes 1, 2 and 5 in other datasets. 

        # While loop ensures only a maximum of third place is selected. 
        while (predictions_one_hot.count(0) <=2):

            # Checks if predicted class and probability is past a threshold. 
            if (prediction == 0) & (predictions_one_hot[0] < 0.4):
                #Sets the probability of this class to 0 and repicks the new most probable class. 
                predictions_one_hot[0] = 0
                prediction = predictions_one_hot.index(max(predictions_one_hot))

            if (prediction == 3) & (predictions_one_hot[3] < 0.85):#/(1+predictions_one_hot.count(0))):
                predictions_one_hot[3] = 0
                prediction = predictions_one_hot.index(max(predictions_one_hot))

            if (prediction == 4) & (predictions_one_hot[4] < 0.75):#/(1+predictions_one_hot.count(0))):
                predictions_one_hot[4] = 0
                prediction = predictions_one_hot.index(max(predictions_one_hot))

            # These can be added in for other datasets if there are too many of the other
                
            # if (prediction == 1) & (predictions_one_hot[1] < 0.2/(1+predictions_one_hot.count(0))):
            #     print("1: ",predictions_one_hot)
            #     predictions_one_hot[1] = 0
            #     prediction = predictions_one_hot.index(max(predictions_one_hot))

            # if (prediction == 5) & (predictions_one_hot[5] < 0.72/(1+predictions_one_hot.count(0))):
            #     print("5: ",predictions_one_hot)
            #     predictions_one_hot[5] = 0
            #     prediction = predictions_one_hot.index(max(predictions_one_hot))

            # if (prediction == 2) & (predictions_one_hot[2] < 0.5/(1+predictions_one_hot.count(0))):
            #     print("2: ",predictions_one_hot)
            #     predictions_one_hot[2] = 0
            #     prediction = predictions_one_hot.index(max(predictions_one_hot))

            break
        
        #Adds to list of Indexes 
        if prediction != 0:
            Indexes.append(peak_start[count])
            Classes.append(prediction)

# Index and Class are saved as a .mat variable in local folder. 
spio.savemat('D2Output.mat', 
                 {'Index': np.array(Indexes),
                  'Class': np.array(Classes)})

print("Completed")
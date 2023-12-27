import numpy as np
from scipy.stats import norm
from scipy.ndimage import median_filter
import scipy.io as spio
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from random import randint

def add_noise(signal, snr_dB):
    """
    Adds Gaussian noise to a 1D signal proportional to a SNR value in dBs. 

    Inputs:
    - signal: list of floats, the original signal
    - snr_dB: integer, desired signal-to-noise ratio of generated noisy signal in decibels

    Outputs:
    - noisy_signal: list of floats, the signal with added noise
    """

    # Calculate the power of the signal and desired noise power from SNR
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = signal_power / (10 ** (snr_dB / 10))

    # Generate Gaussian noise for each value of signal
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Superimpose noise onto original signal 
    noisy_signal = signal + noise
    return noisy_signal

def bandpass_filter_fft(signal, fs, lowcut, highcut):
    """
    Applies a bandpass filter to a signal using FFT and cut off frequencies.

    Inputs:
    - signal: list of floats, original time domain signal
    - fs: integer, the sampling frequency of the signal
    - lowcut: integer, the lower frequency cutoff for the filter
    - highcut: integer, the upper frequency cutoff for the filter

    Outputs:
    - iftt: list of floats, filtered signal
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
    Filters signal using a bandpass and median filter,
    Also creates a noisy version of the original signal.

    Inputs:
    - domain: list of floats, original signal

    Outputs:
    - norm_filt: list of floats, normalised filtered signal
    - norm_noisy: list of floats, normalised noisy signal. 
    """

    # Apply bandpass filter and median filter to the signal to remove high frequency noise.
    freq_filtered = bandpass_filter_fft(domain, 25000, 1, 3000)
    med_filtered = median_filter(freq_filtered, size=3, mode='reflect')

    # Normalise the filtered signal so all values are between 0-1 (for use in CNN).
    Max_filt = max(med_filtered)
    norm_filt = [val/Max_filt for val in med_filtered]

    # Normalise the original signal so all values are between 0-1 (for use in CNN).
    Max_d = max(domain)
    norm_d = [val/Max_d for val in domain]

    # Adds 20dB noise to the normalized original signal.
    snr_dB = 20  
    norm_noisy = add_noise(np.array(norm_d), snr_dB)

    # Normalize the noisy signal to have the same amplitude as the filtered signal
    max_noise = max(norm_noisy)
    norm_noisy = [a/max_noise for a in norm_noisy]

    return (norm_filt, norm_noisy)

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

    # Find the index of the first value that is higher than the mean + 1 standard deviation.
    # The higher values are more likely to belong to peaks. 
    high_peak_cutoff = round(mu + 1.0 * std, 4)

    # Find the index of the first value that is lower than the high peak cutoff based on the sorted lists.
    rev_d = reversed(sorted_d)
    for idx, val in enumerate(rev_d):     
        if round(val,4) == high_peak_cutoff:
            higher_cutoff_index = len(sorted_d) - idx 
            break

    # Takes the indexes of the highest amplitudes in the signal from the cutoff index. 
    # Sorts them in back in order of index.  
    tocheck_indexes = sorted_ind[higher_cutoff_index: len(sorted_d)-1]
    tocheck_indexes = sorted(tocheck_indexes)
    print("Number of values to check: ", len(tocheck_indexes))

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

def generate_spike_data(domain, window_width, window_before, Index):
    """
    Generates windows of spikes to be used as pictures to train a the AI model. 

    Inputs:
    - signal: list of floats, original signal to extract from
    - window_width: integer, width of the spike window
    - window_before: integer, number of indexes before spike peak
    - Index: list of integers, indices of start of spike peak positions

    Outputs:
    - spike_features: array, window pictures of 
    """

    # Calculate number of values after spike
    window_finish = window_width - window_before

    #Initialise list to store spike features.
    spike_features = []

    # Iterate through each spike index
    for point in Index:
        # Extract window of amplitudes after spike start to get whole window. 
        spike_window = domain[point: point + window_width]

        # Find maximum point in window to get peak tip
        peak_max_index = point + spike_window.index(max(spike_window))

        # Extract data surrounding maximum based on desired points before and after. 
        spike = domain[peak_max_index - window_before: peak_max_index + window_finish]

        # Shift the spike up by the DC component of the noise at the first few values to centralise
        # the spike window vertically. 
        DC_Comp = np.mean(spike[0:int(window_before * 0.6)])
        spike = [val - DC_Comp for val in spike]

        # Append window to features list
        spike_features.append(spike)

    print("Imported Labelled Spikes")
    return spike_features

def generate_model(spike_features, Class, one_hot):
    """
    Train a convolutional neural network (CNN) model using spike windows.

    Inputs:
    - spike_features: list of 1D numpy arrays, windows for each spike
    - Class: array-like, labelled classes for each spike
    - one_hot: boolean, selection of one-hot encoding for class labels

    Outputs:
    - model: sequential keras model, trained CNN model
    """

    # Apply one-hot encoding if specified, shifts Class range from 1-5 to 0-4 for use in CNN. 
    if one_hot:
        Class = [a - 1 for a in Class]

    # Convert data to numpy arrays
    spike_features = np.array(spike_features)
    Class = np.array(Class)

    # Split the data into training and testing sets
    # Uses and 80%-20% split respectively.
    X_train, X_test, Y_train, Y_test = train_test_split(spike_features, Class, test_size=0.2, random_state=162)

    # Reshape data to ensure the dimensions are all uniform.
    # Refactors windows into length x 1 arrays and classes into 1x1.  
    X_train = X_train.reshape(X_train.shape[0], window_width, 1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1, 1)
    X_test = X_test.reshape(X_test.shape[0], window_width, 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1, 1)

    # Build CNN using sequential model. Input layer set to width of window. 
    # Consists of 2 convolution/maxpooling layers and a flattening layer. 
    model = models.Sequential()
    model.add(layers.Conv1D(16, 5, activation='relu', input_shape=(window_width, 1)))
    model.add(layers.MaxPooling1D(2, strides=1))
    model.add(layers.Conv1D(8, 5, activation='relu'))
    model.add(layers.MaxPooling1D(2, strides=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))

    # Determine the number of units in the output layer based on one-hot encoding.
    # If one hot encoding, results are 0-4. If not its 0-5. 
    dense_layer = 6 - one_hot
    model.add(layers.Dense(dense_layer, activation='softmax'))  # Output layer with 6 units (0 to 5)

    # Compile the model using Adam optimiser and cross entropy loss function (as classes)
    # don't overlap.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model for 10 epochs. 
    history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
    model.summary()

    # Save the trained model in local folder. 
    model.save("CNN_Model.h5")

    # Displays the results of model evaluation using test data. 
    validation_loss, validation_accuracy = model.evaluate(X_test, Y_test)
    print('Validation loss: ', validation_loss)
    print('Validation accuracy: ', validation_accuracy)

    return model

def generate_window(noisy, actual_peak, window_before, window_finish):
    """
    Generate a window of data about a specified peak in the noisy signal.

    Inputs:
    - noisy: list of floats, noisy time domain signal
    - actual_peak: integer, index of peak in the signal
    - window_before: integer, number of values before the peak
    - window_finish: integer, number of values after the peak

    Outputs:
    - spike: list of integers, window of amplitudes surrounding peak 
    """

    # Create list of surrounding values about peak. 
    spike = noisy[actual_peak - window_before: actual_peak + window_finish]

    # Calculate the mean of the pre-spike window
    Start_DC_Comp = np.mean(spike[0:int(window_before * 0.6)])

    # Shift the signal by the DC component at the start.
    spike = [a - Start_DC_Comp for a in spike]

    return spike


#------------------MAIN FUNCTION--------------------

# Import supplied D1 dataset and split up time domain, Index start location and classes. 
# CODE PROVIDED ON ASSIGNMENT SHEET
mat = spio.loadmat('D1.mat', squeeze_me=True)
domain = mat['d']       # Raw time domain recording, sample rate 25kHz
Index = mat['Index']    # Location in recording, start of each spike
Class = mat['Class']    # Class (1, 2, 3, 4, 5), type of each neuron
Class = list(Class)

# Generate filtered signal and noisy signal. 
(d, noisy) = filter_data(domain)

# Initiate spike window width and number of values before/after peak. 
window_width = 100
window_before = 40
window_finish = window_width-window_before

# Generate spike window pictures for each index to train CNN.
spike_features= generate_spike_data(noisy, window_width, window_before, Index)

# Obtain peaks from the filtered signal in same method for the other noisier datasets
(peaks,max_points) = get_peaks(d)

# Generate a CNN model using one-hot encoding of class values.
model = generate_model(spike_features, Class, 1)

# Intiate a list of predicted indexes and classes based on CNN model.
pred_index = []
pred_class = []

# Iterates through all peak values location from peak detection function
for count, point in enumerate(max_points):
    # Finds window of predicted peak index and reshapes to fit CNN
    actual_peak = point
    check = np.array(d[actual_peak-window_before: actual_peak + window_finish], dtype=np.float32)  
    check = check.reshape(1,window_width, 1)  

    # Predicts the class of peak by selecting the value with the highest probability  
    predictions_one_hot = model.predict(check)
    predictions = np.argmax(predictions_one_hot, axis=1)

    # Saves the corresponding index witht the start of the maximum value and adds 1 bacl to the class
    # to undo one-hot encoding.  
    pred_index.append(peaks[count])
    pred_class.append(predictions[0]+1) #Undo 1 hot encoding        

# Sorts the indexes in ascending order, with associated classes. 
combined_lists = list(zip(Index, Class))
sorted_lists = sorted(combined_lists, key=lambda br: br[0])
sorted_Index, sorted_Class = zip(*sorted_lists)

# MINI AUTOMARKER
# Iterates through all the identified peaks and compares them to the true values provided in the dataset 

incorrect_pos = []      # Stores incorrectly identified indexes
invalid_index = []      # Stores true indexes with a correctedly identified value already assigned. 
retrain_pos = []        # Stores correctly identified peaks but incorrectedly classified. 
retrain_class = []

# Iterates through each predicted index and true index to find a matching one. 
for count_pred, predicted in enumerate(pred_index):
    # Predicted index marked as correct only if a match is found. 
    incorrect = 1
    for count_real, real in enumerate(sorted_Index):
        # If a predicted index is within the 50 point allowance and hasn't already been checked.
        if (abs(predicted-real) <=50) and (real not in invalid_index):
            incorrect = 0

            # Adds index and class to retrain data point list. 
            if pred_class[count_pred] != sorted_Class[count_real]:
                retrain_pos.append(predicted)
                retrain_class.append(sorted_Class[count_real])

            #Associated true values are not checked again
            invalid_index.append(real)
            break
    #If no match, position is assumed incorrectly identified. 
    if incorrect:
        incorrect_pos.append(predicted)
Class = list(Class)

# Creates a window for each incorrectedly identified peak and add to the training dataset with
# a labelled class of 0 (not a peak).
for point in incorrect_pos:
    drange = noisy[point: point+40]
    actual_peak = point + drange.index(max(drange))    
    if (actual_peak > window_before) & (actual_peak < (len(d) - window_finish)):
        flat = generate_window(noisy, actual_peak, window_before, window_finish)
        Class.append(0)
    spike_features.append(flat)

# Repeat for incorrectedly classified peaks, extending the training data by the new window 
# and correctedly associated class. 
for count, point in enumerate(retrain_pos):
    drange = noisy[point: point+40]
    actual_peak = point + drange.index(max(drange))    
    if (actual_peak > window_before) & (actual_peak < (len(d) - window_finish)):
        spike = generate_window(noisy, actual_peak, window_before, window_finish)
        Class.append(retrain_class[count])
    spike_features.append(spike)

# Takes 600 random points along the time domain and generates a window.
# Adds to training set and a class of 0 (not a peak).
for a in range(600):
    random_point = randint(0, len(d))
    if random_point not in Index:
        flat = generate_window(noisy, actual_peak, window_before, window_finish)
        Class.append(0)
        spike_features.append(flat)
print("Imported Non-Spikes")

# Retrain the model the extended dataset without one-hot encoding (as 0 will now mean not a peak)
model = generate_model(spike_features, Class, 0)

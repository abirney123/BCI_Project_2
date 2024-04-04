#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:57:35 2024

@author: Alaina Birney

WRITE TOP LEVEL DESCRIPTION BEFORE SUBMITTING
ONLY WORKS FOR A SSVEP BCI WITH 2 CHOICES
"""
# import necessary libraries
import numpy as np
import math
from matplotlib import pyplot as plt

#%% Part A: Generate Predictions
# load data
def load_ssvep_data(subject, relative_data_path='./SsvepData/'):
    '''
    Function to load in the .npz data file containing EEG data for a given 
    subject for SSVEP analysis. Please note that this function was originally 
    written with Ron Bryant for lab 3.
    
    Parameters
    ----------
    subject : Int, required.
        The number of the subject for whom data will be loaded.
    relative_data_path : Str, optional
        The relative path to the data files. The default is './SsvepData/'.

    Returns
    -------
    data_dict : numpy.lib.npyio.NpzFile of size f where f represents the number
    of arrays within this object. 
        This object behaves similarly to a dictionary and can be accessed using 
        keys. Contains raw, unfiltered information about the dataset. Each 
        array within the dictionary holds information corresponding to a 
        different field. In our case, f=6 and corresponds to: eeg data in volts 
        ("EEG"), channels ("channels"), sampling frequency in Hz ("fs"), the 
        sample when each event occured ("event_samples"), the event durations 
        ("event_durations"), and the event types ("event_types").
        

    '''
    # Load dictionary
    data_dict = np.load(f'{relative_data_path}SSVEP_S{subject}.npz',
                        allow_pickle=True)
    return data_dict

# extract epochs with custom start and end times
def epoch_ssvep_data(data_dict, freq_b, epoch_start_time=0, epoch_end_time=20):
    '''
    A function to extract epochs aand produce a variable to represent the time 
    of each sample in each epoch, relative to the event onset. Epochs begin 
    when event samples start and end 20 seconds later by default. Please note 
    that this function was originally written with Ron Bryant for lab 3 and 
    modified to be more flexible in terms of what frequency can be detected 
    to produce an array indicating whether or not each epoch corresponded to 
    that frequency. This modification included the addition of the input 
    parameter "freq_b". The function was also modified to be able to 
    handle epoch start and end times that are not whole numbers.

    Parameters
    ----------
    data_dict : numpy.lib.npyio.NpzFile of size F where F represents the number of 
    arrays within this object. Required.
        This object behaves similarly to a dictionary and can be accessed using 
        keys. Contains raw, unfiltered information about the dataset. Each 
        array within the dictionary holds information corresponding to a 
        different field. In our case, F=6 and corresponds to: eeg data in volts 
        ("EEG"), channels ("channels"), sampling frequency in Hz ("fs"), the 
        sample when each event occured ("event_samples"), the event durations 
        ("event_durations"), and the event types ("event_types").
    freq_b : Int
        The frequency for which data will be evaluated to see if the light was
        flashing at that frequency for each epoch. 
    epoch_start_time : Int, optional
        The time that each epoch begins in seconds, relative to the event 
        sample. The default is 0.
    epoch_end_time : Int, optional
        The time that each epoch ends in seconds, relative to the event sample.
        The default is 20.

    Returns
    -------
    eeg_epochs : Array of float. Size (E,C,T) where E is the number of epochs,
    C is the number of EEG channels, and T is time points.
        EEG data in uV.
    epoch_times : Array of float. Size (T,) where T is time points.
        The time in seconds of each time point in eeg_epochs, relative to the 
        event.
    is_trial_bHz : Array of bool. Size (E,) where E is the number of epochs.
        An indication of whether the light was flashing at frequency b during 
        each epoch. True if the light was flashing, false otherwise.

    '''
    
    #epoch_start_time = int(epoch_start_time) # math.floor(epoch_start_time*10)/10
    #epoch_end_time = int(epoch_end_time) #math.floor(epoch_end_time*10)/10
    

    #unpack data_dict
    eeg_data = data_dict['eeg']/1e-6   # convert to microvolts
    fs = data_dict['fs']                # sampling frequency
    event_samples = data_dict['event_samples']    #index to start of events
    event_types = data_dict['event_types']    # image frequency of during event
    
    # calculate epoch parameters
    epoch_start_indexes = (event_samples + np.round(epoch_start_time * fs)).astype(int)
    epoch_end_indexes = (event_samples + np.round(epoch_end_time * fs)).astype(int)
    epoch_durations = int((epoch_end_time - epoch_start_time) * fs) # duration in samples
    epoch_times = np.arange(epoch_start_time, epoch_end_time, 1/fs)  #seconds
    epoch_count = len(event_samples)
    
    #initaialize variables to store epochs and indication of whether b Hz trial
    eeg_epochs = np.zeros((epoch_count,
                           eeg_data.shape[0],
                           epoch_durations )) 
    is_trial_bHz = np.zeros(epoch_count, dtype=bool)
    
    # convert freq_b to string matching necessary format
    freq_b = str(freq_b) + "hz"
    
    #populate
    for event_index in range(epoch_count):
        start_eeg_index = epoch_start_indexes[event_index]
        stop_eeg_index = epoch_end_indexes[event_index]
        eeg_epochs[event_index,:,:stop_eeg_index-start_eeg_index]  \
                = eeg_data[:,start_eeg_index:stop_eeg_index]
        is_trial_bHz[event_index] =   event_types[event_index] == freq_b
    
    return eeg_epochs, epoch_times, is_trial_bHz

# take Fast Fourier Transform (FFT) of each epoch to examine frequency content
def get_frequency_spectrum(eeg_epochs, fs):
    '''
    A function to calculate the Fourier transform on each channel in each epoch.
    Please note that this function was originally written with Ron Bryant for 
    lab 3, but was modified for this project so that it no longer includes an 
    option to remove the DC offset. This modification was made for simplicity.
    
    Parameters
    ----------
    eeg_epochs : Array of float. Size (E,C,T) where E is the number of epochs,
    C is the number of EEG channels, and T is time points. Required.
        EEG data in uV.
    fs : Int, required.
        The sampling frequency in Hz.

    Returns
    -------
    eeg_epochs_fft : Array of complex float. Size (E,C,F) where E is the number 
    of epochs, C is the number of EEG channels, and F is the number of frequencies.
    The number of frequencies is equal to (number of time points/2)+1 when the 
    number of time points is even (as it is in our data) and is equal to 
    (number of time points +1)/2 if the number of time points is odd.
        Fourier transformed EEG epochs. Values represent the frequency spectra.
    fft_frequencies : Array of float. Size (F,) where F is the number of 
    frequencies. 
        Frequencies corresponding to columns in eeg_epochs_fft such that a 
        frequency spectrum value within eeg_epochs in column i corresponds to
        the frequency in fft_frequencies at row i. Frequencies range from 0 to
        fs/2 where fs is the sampling frequency, in accordance with the Nyquist 
        Criterion.

    '''
            
    # perform Fourier transform on each channel in each epoch
    eeg_epochs_fft = np.fft.rfft(eeg_epochs, axis=-1)
    
    # get corresponding frequencies
    # d represents sample spacing (inverse of sample rate)
    fft_frequencies = np.fft.rfftfreq(eeg_epochs.shape[2], d=1/fs)
    
    return eeg_epochs_fft, fft_frequencies


def find_frequency_indices(fft_frequencies, freq_a, freq_b):
    """
    A function to find the indices of the of the FFT eeg data that align with 
    frequencies shown in each event. If no matching frequencies are found within
    the FFT data, the indices corresponding to the closest frequencies will be 
    chosen.
    
    Parameters
    ----------
    fft_frequencies : Array of float. Size (F,) where F is the number of frequencies.
    The number of frequencies is equal to (number of time points in EEG data / 2 +1)
    when the number of time points is even and is equal to (number of time points 
    in EEG data +1) / 2 if the number of time points is odd.
        Fast Fourier transformed EEG data. Values represent the frequency spectra.
    freq_a : Int
        One of the frequencies for the flashing shown in the SSVEP experiment.
    freq_b : Int
        The other frequency of the flashing shown in the SSVEP experiment.

    Returns
    -------
    fft_idx_freq_a : Int
        The index of fft_frequencies where the value is closest to freq_a.
    fft_idx_freq_b : Int
        The index of fft_frequencies where the value is closest to freq_b.

    """
    # find fft indices for frequency a
    # get abs value of difference between all fft frequencies and frequency a
    freq_a_fft_dif = np.abs(fft_frequencies - freq_a)
    
    # add smallest values to variable to store fft index for frequency a
    fft_idx_freq_a = np.argmin(freq_a_fft_dif)
    
    # find fft indices for frequency b
    # get abs value of difference between all fft frequencies and frequency b
    freq_b_fft_dif = np.abs(fft_frequencies - freq_b)
    
    # add smallest values to variable to store fft index for frequency b
    fft_idx_freq_b = np.argmin(freq_b_fft_dif)
    
    return fft_idx_freq_a, fft_idx_freq_b

def generate_prediction(eeg_epochs_fft, fft_idx_freq_a, fft_idx_freq_b,
                        electrode, channels, freq_a, freq_b):
    """
    A function to predict the stimulus frequency that the subject was focused on
    during each epoch. The prediction is based on which frequency (frequency a 
    or frequency b) was associated with a higher amplitude. 
    
    Parameters
    ---------
    eeg_epochs_fft : Array of complex float. Size (E,C,F) where E is the number 
    of epochs, C is the number of EEG channels, and F is the number of frequencies.
    The number of frequencies is equal to (number of time points/2)+1 when the 
    number of time points is even (as it is in our data) and is equal to 
    (number of time points +1)/2 if the number of time points is odd.
        Fourier transformed EEG epochs. Values represent the frequency spectra.
    fft_idx_freq_a : Int
        The index of fft_frequencies where the value is closest to freq_a.
    fft_idx_freq_b : Int
        The index of fft_frequencies where the value is closest to freq_b.
    electrode : Str
        The desired electrode. Predictions will be made for data within the 
        corresponding channel.
    channels : Array of str. Size (C,) where C is the number of channels for
    which we have data.
        Channel names. Indices of this array correspond to columns within 
        eeg_epochs_fft, meaning the index corresponding to an electrode or 
        channel name within channels will match the column index for FFT EEG 
        data for that channel within eeg_epochs_fft.
    freq_a : Int
        One of the frequencies for the flashing shown in the SSVEP experiment.
    freq_b : Int
        The other frequency of the flashing shown in the SSVEP experiment.
        
    Returns
    -------
    predictions : List of size (E,) where E is the number of epochs in 
    eeg_epochs_fft.
        The predictions of the stimulus frequency for each epoch.

    """
    # get channel index from electrode name
    # loop through channels, values are strings corresponding to electrodes
    # while indices correspond to columns of eeg_epochs_fft
    channel_idx = None
    for channel_index, channel_value in enumerate(channels):
            if channel_value == electrode:
                channel_idx = channel_index
                break
    
    if channel_idx is not None:
        # extract amplitudes at the fft indices corresponding to frequency a and 
        # frequency b at the desired electrode by indexing eeg_epochs_fft
        # and taking absolute value
        amplitude_a = np.abs(eeg_epochs_fft[:, channel_idx, fft_idx_freq_a])
        amplitude_b = np.abs(eeg_epochs_fft[:, channel_idx, fft_idx_freq_b])
        
        # find which amplitude is higher by calculating difference between average 
        #amplitude a and average amplitude b. If positive, amplitude a is higher on 
        #average. If negative, amplitude b is higher on average. If zero, they are 
        #equal. amplitude_differences will contain one value for each epoch
        amplitude_differences = amplitude_a - amplitude_b
        
        # initialize list to store predictions
        predictions = []
        
        # set variable to keep track of epochs
        epoch = 1
        
        # loop through amplitude differences, evaluate, and store frequency corresponding
        # to the higher amplitude in the predictions list
        for difference in amplitude_differences:
            if difference > 0:
                predictions.append(freq_a)
            elif difference < 0:
                predictions.append(freq_b)
            elif difference == 0:
                predictions.append(None)
                print("The average amplitudes for the given frequencies are equal, therefore" +
                  f" a meaningful prediction cannot be made for epoch {epoch}, corresponding "+
                  f" to index {epoch-1} in the predictions list.")
            # increment epoch tracker
            epoch += 1
            
        return predictions
    else:
        print("The desired electrode could not be found. Please ensure the electrode "+
              "name was spelled correctly and is contained within the list of channels" +
              "for the SSVEP data.")
        
#%% Part B: Calculate Accuracy and ITR
def calculate_accuracy(is_trial_bHz, predictions, freq_b):
    """
    A function to calculate the accuracy of the predictions.
    
    Parameters
    ----------
    is_trial_bHz : Array of bool. Size (E,) where E is the number of epochs.
        An indication of whether the light was flashing at frequency b during 
        each epoch. True if the light was flashing, false otherwise.
    predictions : List of size (E,) where E is the number of epochs in 
    eeg_epochs_fft.
        The predictions of the stimulus frequency for each epoch.
    freq_b : Int
        One of the frequencies of the flashing shown in the SSVEP experiment.

    Returns
    -------
    accuracy : Float
        The accuracy of the predictions of trial type. Calculating by dividing 
        the number of accurate predictions by the total number of predictions.

    """

    # is_trial_bhz has T/F values, predictions has ints. Create variable to 
    # store predictions as T/F values relative to if bHz for comparison
    is_prediction_bHz = [] # initialize list to store T/F values
    for prediction in predictions:
        if prediction == freq_b:
            is_prediction_bHz.append(True)
        elif prediction == None: # for cases where difference in amplitude
        # was 0. User would have been notified prior so no additional messaging
        # needed.
            is_prediction_bHz.append(None)
        elif (prediction != None) & (prediction != freq_b):
            is_prediction_bHz.append(False)
    
    # accuracy = number of correct predictions / number of total predictions
    num_correct_predictions = is_prediction_bHz == is_trial_bHz
    accuracy = num_correct_predictions.sum() / len(predictions)
    
    return accuracy

def get_ITR(accuracy, epoch_start_time = 0, epoch_end_time = 20, num_choices = 2):
    """
    A function to calculate the ITR of the predictions in bits per second.

    Parameters
    ----------
    accuracy : Float
        The accuracy of the predictions of trial type.
    epoch_start_time : Int
        The time that each epoch begins in seconds, relative to the event 
        sample. The default is 0.
    epoch_end_time : Int, optional
        The time that each epoch ends in seconds, relative to the event sample.
        The default is 20.
    num_choices : Int, optional
        The number of stimuli presented to the subject during the SSVEP experiment.
        The default is 2.

    Returns
    -------
    ITR_time : Float
        The information transfer rate in bits per second.

    """
    
    # get trials per second, epoch start and end times are in seconds
    epoch_duration = epoch_end_time - epoch_start_time
    trials_per_second = 1/epoch_duration
    num_choices = 2
    
    # get ITR(trial)
    # ITR(trial) = log2N + P*log2P + (1-P) * log2([1-P]/[N-1])
    # where N is number of trials and P is accuracy
    # handle cases where accuracy is 0 or 1 to avoid invalid values when taking log
    if accuracy == 0:
        ITR_trial = np.log2(num_choices) + (1-accuracy) * np.log2((1-accuracy)/(num_choices - 1))
    elif accuracy == 1: 
        ITR_trial = np.log2(num_choices) + (accuracy) * np.log2(accuracy) 
    else:
        ITR_trial = np.log2(num_choices) + (accuracy) * np.log2(accuracy) + (1-accuracy) * np.log2((1-accuracy)/(num_choices - 1))
    # get ITR(time)
    # ITR(time) = ITR(trial) * (trails/ sec)
    ITR_time = ITR_trial * (trials_per_second)
    return ITR_time


#%% Part C: Loop Through Epoch Limits
def test_epochs(data_dict, epoch_start_times, epoch_end_times, freq_a,
                freq_b, subject, electrode, num_choices = 2):
    """
    A function to test various epoch start and end times for SSVEP data. For 
    each valid combination (a valid combination occurs when the epoch end time
    is greater than the start time) of the epoch start and end times provided 
    as an input, this function separates EEG data into epochs, calculates FFTs,
    generates predictions of the stimulus frequency, and evaluates the performance 
    of these predictions through accuracy and Information Transfer Rate (ITR). 
    
    Parameters:
    ----------
    data_dict : numpy.lib.npyio.NpzFile of size f where f represents the number
    of arrays within this object. 
        This object behaves similarly to a dictionary and can be accessed using 
        keys. Contains raw, unfiltered information about the dataset. Each 
        array within the dictionary holds information corresponding to a 
        different field. In our case, f=6 and corresponds to: eeg data in volts 
        ("EEG"), channels ("channels"), sampling frequency in Hz ("fs"), the 
        sample when each event occured ("event_samples"), the event durations 
        ("event_durations"), and the event types ("event_types").
    epoch_start_times: List of size (s,) where s is the number of start times 
    to be tested.
        Start times (in seconds) to test for epoching the data.
    epoch_end_times: List of size (e,) where e is the number of end times
    to be tested.
        End times (in seconds) to test for epoching the data.
    freq_a : Int
        One of the frequencies of the flashing shown in the SSVEP experiment.
    freq_b : Int
        The other frequency of the flashing shown in the SSVEP experiment.
    subject : Int
        The number of the subject for whom data will be loaded.
    electrode : Str
        The desired electrode. Predictions will be made for data within the 
        corresponding channel.
    num_choices : Int, optional
        The number of stimuli presented to the subject during the SSVEP experiment.
        The default is 2.
    
    Returns:
    -------
    results : Dict of size N where N represents the number of valid combinations
    of start and end times within epoch_start_times and epoch_end_times.
        The results of the testing. The keys are tuples representing 
        the epoch window defined by the start and end times. Each value in the 
        dictionary is another dictionary with the following keys:
        - "accuracy": The accuracy of the predictions made for the epoch window.
        - "ITR": The Information Transfer Rate (ITR) in bits per second for the 
        predictions made for the epoch window.
    """
    # initialize list to store results
    results = {}
    
    # loop through start and end times
    for start_time in epoch_start_times:
        for end_time in epoch_end_times:
            # don't evaluate cases where start time > end time
            if end_time > start_time:

                # skip if not whole number
                #if (start_time % 1 != 0 or end_time % 1 != 0):
                    #continue

                # epoch data
                eeg_epochs, epoch_times, is_trial_bHz = epoch_ssvep_data(data_dict,
                                                                         freq_b,
                                                                         epoch_start_time=start_time,
                                                                         epoch_end_time=end_time)
                # calculate FFT
                fs = data_dict["fs"] # get fs to use as input for getting frequency spectrum
    
                # Skip due to time interval being too small
                if (eeg_epochs.size == 0):
                    continue

                eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
                
                # get frequency indices for prediction generation
                fft_idx_freq_a, fft_idx_freq_b = find_frequency_indices(fft_frequencies,
                                                                        freq_a, freq_b)
                # generate predictions
                channels = data_dict["channels"] # get channels to use as input for predictions
                predictions = generate_prediction(eeg_epochs_fft, fft_idx_freq_a,
                                                  fft_idx_freq_b, electrode,
                                                  channels, freq_a, freq_b)
                # calculate accuracy
                accuracy = calculate_accuracy(is_trial_bHz, predictions, freq_b)
                
                
                # calculate ITR
                ITR = get_ITR(accuracy, epoch_start_time = start_time,
                              epoch_end_time = end_time, num_choices = num_choices)
                
                # save info to dictionary
                results[(start_time, end_time)] = {"accuracy" : accuracy,
                                                   "ITR" : ITR}
                
                
    return results
                
#%% Part D: Plot Results
def generate_pseudocolor_plots(results, epoch_start_times, epoch_end_times, subject):
    """
    A function to generate and save pseudocolor plots to evaluate the accuracies 
    and ITRs at various epoch limits.

    Parameters
    ----------
    results : results : Dict of size N where N represents the number of valid combinations
    of start and end times within epoch_start_times and epoch_end_times.
        The results of the testing. The keys are tuples representing 
        the epoch window defined by the start and end times. Each value in the 
        dictionary is another dictionary with the following keys:
        - "accuracy": The accuracy of the predictions made for the epoch window.
        - "ITR": The Information Transfer Rate (ITR) in bits per second for the 
        predictions made for the epoch window.
    epoch_start_times: List of size (s,) where s is the number of start times 
    to be tested.
        Start times (in seconds) to test for epoching the data.
    epoch_end_times: List of size (e,) where e is the number of end times
    to be tested.
        End times (in seconds) to test for epoching the data.
    subject : Int
        The number of the subject for whom data will be plotted. This is only
        used for plot labels, but must match the subject number fed to other
        functions for loading and epoching data to ensure the plot title is
        accurate.

    Returns
    -------
    None.

    """
    # get accuracy and ITR for each epoch
    # initialize list to store all accuracies
    
    accuracies = np.zeros((len(epoch_start_times), len(epoch_end_times)))
    # initialize list to store all ITRs
    ITRs = np.zeros((len(epoch_start_times), len(epoch_end_times)))
    
    # loop through keys, get accuracy and ITR for each start, end pair
    for start_idx, start_time in enumerate(epoch_start_times):
        for end_idx, end_time in enumerate(epoch_end_times):
            key = (start_time, end_time)
            if key in results:
                accuracies[start_idx, end_idx] = results[key]["accuracy"]
                ITRs[start_idx, end_idx] = results[key]["ITR"]
                
    # convert accuracies to percentages
    accuracies = accuracies * 100
    
    # pseudocolor plot for accuracy - color is accuracy, x is end time, y is start time   
    plt.figure()
    plt.pcolor(epoch_end_times, epoch_start_times, accuracies)
    plt.colorbar(label="% correct")
    plt.ylabel("Epoch Start Time (s)")
    plt.xlabel("Epoch End Time (s)")
    plt.xticks(ticks=np.arange(0,max(epoch_end_times)+1,5))
    plt.yticks(ticks=np.arange(0,max(epoch_start_times)+1,2.5))
    plt.title("Accuracy")
    plt.tight_layout()
    plt.show()
    # save
    filename = f"Accuracy_pseudocolor_s{subject}.png"
    plt.savefig(filename)
    

    # pseudocolor plot for ITR - color is ITR, x is end time, y is start time   
    plt.figure()
    plt.pcolor(epoch_end_times, epoch_start_times, ITRs)
    plt.colorbar(label="ITR (bits/sec")
    plt.ylabel("Epoch Start Time (s)")
    plt.xlabel("Epoch End Time (s)")
    plt.xticks(ticks=np.arange(0,max(epoch_end_times)+1,2.5))
    plt.yticks(ticks=np.arange(0,max(epoch_start_times)+1,5))
    plt.title("Information Transfer Rate")
    plt.tight_layout()
    plt.show()
    # save
    filename = f"ITR_pseudocolor_s{subject}.png"
    plt.savefig(filename)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:57:35 2024

@author: Alaina Birney

WRITE TOP LEVEL DESCRIPTION BEFORE SUBMITTING
"""
# import necessary libraries
import numpy as np

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
    A function to extract epochs around each event and produce a variable to 
    represent the time of each sample in each epoch, relative to the event 
    onset. Epochs begin when event samples start and end 20 seconds later by
    default. Please note that this function was originally written with Ron 
    Bryant for lab 3 and modified to be more flexible in terms of what frequency
    can be detected to produce an array indicating whether or not each epoch 
    corresponded to that frequency. This modification included the addition of
    the input parameter "freq_b".

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
    #unpack data_dict
    eeg_data = data_dict['eeg']/1e-6   # convert to microvolts
    fs = data_dict['fs']                # sampling frequency
    event_samples = data_dict['event_samples']    #index to start of events
    event_types = data_dict['event_types']    # image frequency of during event
    
    # calculate epoch parameters
    epoch_start_indexes = event_samples + int(np.round(epoch_start_time * fs))
    epoch_durations = int(np.round((epoch_end_time - epoch_start_time) * fs))
    epoch_times = np.arange(epoch_start_time, epoch_end_time, 1/fs)  #seconds
    epoch_count = len(event_samples)
    
    #initaialize variables to store epochs and indication of whether 15 Hz trial
    eeg_epochs = np.zeros( ( epoch_count, 
                             eeg_data.shape[0], 
                             len(epoch_times) )
                         ) 
    is_trial_bHz = np.zeros(epoch_count, dtype=bool)
    
    # convert freq_b to string matching necessary format
    freq_b = str(freq_b) + " Hz"
    
    #populate
    for event_index in range(0, len(event_samples)):
        start_eeg_index = epoch_start_indexes[event_index]
        stop_eeg_index = start_eeg_index + epoch_durations
        eeg_epochs[event_index,:,:]  \
                = eeg_data[:,start_eeg_index:stop_eeg_index]
        is_trial_bHz[event_index] =   event_types[event_index] == freq_b
        
    return eeg_epochs, epoch_times, is_trial_bHz

# take Fast Fourier Transform (FFT) of each epoch to examine frequency content
def get_frequency_spectrum(eeg_epochs, fs):
    '''
    A function to calculate the Fourier transform on each channel in each epoch.
    An optional parameter, remove_DC has been added to allow users to indicate 
    whether they would like to remove the DC offset from EEG signal. This can 
    be useful because as the DC offset is anmartifacts and may be large which 
    will unnecessarily decrease the normalized power of the signal. Setting 
    this variable to False retains the DC offset. Please note that this function 
    was originally written with Ron Bryant for lab 3.
    
    Parameters
    ----------
    eeg_epochs : Array of float. Size (E,C,T) where E is the number of epochs,
    C is the number of EEG channels, and T is time points. Required.
        EEG data in uV.
    fs : Int, required.
        The sampling frequency in Hz.
    remove_DC: Bool, optional.
        An indication of whether or not to remove the DC offset. The default is 
        False.

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
    frequencies shown in each epoch. If no matching frequencies are found within
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
    A function to predict the stimulus frequency that was shown during each epoch.
    The prediction is based on which frequency (frequency a or frequency b) was 
    associated with a higher amplitude. 
    
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
        amplitude_a = eeg_epochs_fft[:, channel_idx, fft_idx_freq_a]
        amplitude_b = eeg_epochs_fft[:, channel_idx, fft_idx_freq_b]
        
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
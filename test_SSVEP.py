#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:57:37 2024

@author:
    Alaina Birney
    Arthur Dolimier

A python module designed to aid in investigation of which epoch start and end
times are best to use for each subject in an SSVEP experiment as well as which 
threshold should be used for predictions regarding which stimulus the subject
was focused on. This module is suitable for use to analyze data from an SSVEP 
experiment with two choices. These tasks are accomplished by calling various
functions from the SSVEP module and gathering results for various epoch start
and end times.
"""
# import module
import SSVEP as SSVEP
import numpy as np
#%% Part A: Generate Predictions

# set necessary variables
# freq_a and freq_b : the frequencies used in trials of the SSVEP experiment
freq_a = 12
freq_b = 15

# epoch start and end times
epoch_start_time = 0
epoch_end_time = 20

# desired electrode
electrode = "Oz" # Oz was chosen for this SSVEP experiment because it is located
# over the occipital lobe 

# subject
subject = 2

# number of choices in SSVEP experiment- must be 2
num_choices = 2

# load data
data_dict = SSVEP.load_ssvep_data(subject, relative_data_path='./SsvepData/')
# define fs
fs = data_dict["fs"]
# define channels
channels = data_dict["channels"]

# get eeg epochs, epoch times, and an indication of trial type
eeg_epochs, epoch_times, is_trial_bHz = SSVEP.epoch_ssvep_data(data_dict, freq_b,
                                                               epoch_start_time,
                                                               epoch_end_time)

# take Fast Fourier Transform (FFT) of each epoch to examine frequency content
# subject 1
eeg_epochs_fft, fft_frequencies = SSVEP.get_frequency_spectrum(eeg_epochs, fs)

# find the elements of the FFT representing the amplitude of oscillations
# at the two stimulus frequencies
fft_idx_freq_a, fft_idx_freq_b = SSVEP.find_frequency_indices(fft_frequencies,
                                                                      freq_a, freq_b)
# get predictions for each epoch
predictions = SSVEP.generate_prediction(eeg_epochs_fft, fft_idx_freq_a,
                                           fft_idx_freq_b, electrode,
                                           channels, freq_a, freq_b)

#%% Part B: Calculate Accuracy and ITR
# get accuracy
accuracy = SSVEP.calculate_accuracy(is_trial_bHz, predictions, freq_b)

# get ITR in bits per second
ITR_time = SSVEP.get_ITR(accuracy)

#%% Part C: Loop Through Epoch Limits

# set possible epoch start and end times
# have start range from 0 to 19 so it is within the bounds of the actual stimulus
epoch_start_times = np.arange(0,20,1)
# have end range from 1 to 20 so it is within the bounds of the actual stimulus
epoch_end_times = np.arange(1,21,1)

# calculate figures of merit
results = SSVEP.test_epochs(data_dict, epoch_start_times, epoch_end_times,
                            freq_a, freq_b, subject, electrode, num_choices)

# print(results)

#%% Part D: Plot Results
SSVEP.generate_pseudocolor_plots(results, epoch_start_times, epoch_end_times,
                                 subject, electrode)

# Part E: Predictor Histogram
SSVEP.plot_predictor_histogram(data_dict, 17, 18, freq_a, freq_b, channels, electrode, subject)




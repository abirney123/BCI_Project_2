#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:57:37 2024

@author: Alaina Birney

WRITE TOP LEVEL DESCRIPTION BEFORE SUBMITTING
"""
# import module
import SSVEP as SSVEP
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
subject = 1

# load data
# subject 1
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

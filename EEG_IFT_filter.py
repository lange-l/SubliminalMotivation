"""
author: Lena Lange
last update: 16.01.2025
Python 3.13

Pre-processing of EEG data for replication of Pessiglione et al. (2007) using unmasked stimuli
INCENTIVE FORCE TASK

Pre-processing steps:
- light filtering (50 Hz Notch, 1-100 Hz Bandpass) of every recording block individually
- interpolating bad channels (marked manually)
- re-reference to average
- downsample to 512 Hz
- final bandpass (1-45 Hz)

main001:
main002:
main003:
main004: recording crashed during IFT, new recording for last 3 blocks in main004B --> introduce a bad boundary!
main005: MATLAB script crashed after 180 IFT trials, completed 180 more --> introduce a bad boundary!
main006: recording crashed after 2 IFT blocks (006A), behav. data for last 4 blocks (006B) lost
main007: restarted IFT task
main008:
main009:
main0010:
main0011: behav. data lost (Arduino amplifier died during IFT)
main0012: IFT in 012B; 6th IFT block too long, cut after 270 trials
main0013: no EEG (malfunctioning electrodes)
main0014: IFT terminated before 270 trials (red filling error); most sections EEG doesn't work
main0015: recording crashed during IFT, behav. data lost

"""

import mne # v1.8.0
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # TkAgg, Agg, Qt5Agg

#####################################################################################
### Custom functions

event_IDs = {
    'eegON' : 200,
    'eegOFF' : 199,
    'scrON' : 99,
    'rest_start' : 20,
    'rest_end' : 21,
    'max_start' : 22,
    'max_end' : 23,
    'practice_trial_start' : 24,
    'practice_trial_end' : 25,
    'readVoltage_start' : 33,
    'readVoltage_end' : 34,
    'start_ift' : 44,
    'end_ift' : 45,
    'start_ift_practice' : 46,
    'end_ift_practice' : 47,
    'start_pt' : 55,
    'end_pt' : 56,
    'start_pt_practice' : 57,
    'end_pt_practice' : 58,
    'cent_diff1' : 61,
    'cent_diff2' : 62,
    'cent_diff3' : 63,
    'euro_diff1' : 64,
    'euro_diff2' : 65,
    'euro_diff3' : 66,
    'fix_cross_on' : 71,
    'stim_window_on' : 72,
    'thermo_on' : 73,
    'thermo_off' : 74,
    'reward_on' : 75,
    'reward_off' : 76,
    'pas' : 80,
    'pas_answer' : 81} # trigger reference table

def set_channels(data_path, file_name, ch_type, ch_drop, ch_loc):
    """
    Load raw, continuous EEG data in .fif format, set channel types, drop unwanted channels, apply standard montage to 
    set channel locations.
    
    Parameters
    ----------
    data_path : str
        The path to the directory containing the EEG data file.
    file_name : str
        The name of the .fif file to load.
    ch_type : dict
        A dictionary specifying the channel types to set, where keys are channel names (str)
        and values are channel types (str), e.g., {'EOG1': 'eog'}.
    ch_drop : list of str
        A list of channel names to drop from the dataset.
    ch_loc : str
        The name of the standard montage to apply, e.g., 'biosemi64'.
        
    Returns
    -------
    raw : mne.io.Raw
        The raw EEG data with channel information.
    """

    # read and load raw data
    raw = mne.io.read_raw_fif(data_path + file_name, preload=True)

    # set channel types
    raw.set_channel_types(ch_type)

    # drop unused channels
    raw.drop_channels(ch_drop)

    # set sensor locations with standard montage
    montage = mne.channels.make_standard_montage(ch_loc)
    raw.set_montage(montage)

    return raw

def segment_and_filter(raw, cross_trigger, n_blocks, n_freqs, l_freq, h_freq):
    """
    Segment the continuous EEG file into same-sized blocks by counting trials using the event array (e.g., using the
    fix. cross trigger), apply light filters

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous raw EEG data file (e.g., .fif).
    cross_trigger : int
        Trigger ID of the fixation cross
    n_blocks : int
        The number of blocks in the recording.
    n_freqs : list of int
        A list of frequencies for the notch filter.
    l_freq : float
        Frequency for high-pass filter.
    h_freq : float
        Frequency for low-pass filter.

    Returns
    -------
    raw_filt : mne.io.EEG
        Continuous pre-filtered EEG data
    """

    # find events
    events = mne.find_events(raw) # returns 2d array

    # filter out fix. cross events, adjust trigger times to align with the loaded file (not the uncropped original)
    trials = events[events[:, 2] == cross_trigger] # create 2d array with all events that match the desired trigger ID
    trials[:, 0] -= raw.first_samp

    # check if participant completed 270 trials
    if len(trials) == 270:

        # split into 6 equally sized arrays (i.e., 6 blocks à 45 trials each)
        blocks = np.array_split(trials, n_blocks)  # list of 6 arrays [45, 3]

        # get start and end times of every block
        start_samples = [block[0, 0] for block in blocks]
        end_samples = [block[-1, 0] for block in blocks]
        sfreq = raw.info['sfreq']
        start_times = [sample / sfreq for sample in start_samples]
        end_times = [sample / sfreq for sample in end_samples]

        # cut EEG recording into 6 segements
        eeg_blocks = [raw.copy().crop(tmin=start, tmax=end) for start, end in zip(start_times, end_times)]

        # apply notch filter and light bandpass filters to each block individually
        filtered_blocks = []
        for i, block in enumerate(eeg_blocks, start=1):
            print(f"Applying filters to block {i} ({id})...")
            block.notch_filter(freqs=n_freqs, picks="eeg", method="spectrum_fit")
            block.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", pad="reflect_limited")
            filtered_blocks.append(block)

        # re-concatenate filtered blocks
        print("Re-concatenating filtered blocks...")
        raw_filt = mne.concatenate_raws(filtered_blocks)

        raw_dict[id] = raw_filt
        raw_filt.save(dir_prepro + fname_filt % id, overwrite=True)

        return raw_filt

    else:
        print(f"Irregular number of trials for {raw}")

        return None

def preprocess(raw_filt):
    """
    Plot pre-filtered data to mark bad channels, interpolate, re-reference to average, downsample to 512 Hz, apply final
    bandpass filter (1 to 45 Hz)

    Parameters
    ----------
    raw_filt : mne.io.EEG
        continuous pre-filtered EEG data
    data_path : str
        The path to save the pre-processed EEG file to.
    file_name : str
        The name to save the pre-processed EEG file as.

    Returns
    -------
    raw_prepro : mne.io.EEG
        Continuous pre-processed EEG data
    """

    # plot filtered data & mark bad channels
    events = mne.find_events(raw_filt)
    raw_filt.plot(events=events, picks='eeg', n_channels=64, duration=35)
    plt.show(block=True)  # script continues after plot is closed

    # exclude and interpolate bad channels
    print(f'Interpolating participant {id}...')
    raw_prepro = raw_filt.copy()
    raw_prepro.interpolate_bads(reset_bads=True)

    # re-reference to average
    raw_prepro.set_eeg_reference(ref_channels='average', ch_type='eeg')

    # downsample to 512 Hz (from 2048 Hz)
    print(f'Resampling participant {id}...')
    raw_prepro.resample(512, npad='auto')

    # apply stricter lowpass filter (FIR, 45 Hz)
    print(f'Applying final filters to participant {id}...')
    raw_prepro.filter(l_freq=1.0, h_freq=45.0, picks='eeg')

    return raw_prepro

#####################################################################################
### Read and load raw EEG data, set channels information, apply light filters for identifying bad channels more easily

# data path
dir_prepro = 'C:/Users/Admin/Documents/PessRep_eeg_data/prepro/'

fname_ift = r'%s_IFT_raw.fif' # raw EEG data, cropped to main IFT
fname_filt = r'%s_IFT_filt_eeg.fif' # pre-filtered EEG data
fname_prepro = r'%s_IFT_prepro_eeg.fif' # pre-processed EEG data

# participants
ID_list = ['main001', 'main002', 'main003', 'main004', 'main005', 'main006', 'main007',
           'main008', 'main009', 'main010', 'main011', 'main012', 'main013', 'main014', 'main015']
ID_excl = [
    'main005', # how do behavioural and EEG trials line up???
    'main006', # only 90 trials of behav. data saved
    'main011', # behav. data lost (Arduino Amp died)
    'main013', # EEG did not work
    'main014', # only short sections of EEG data
    'main015' # behav. data lost (red filling error)
]

# channel information
ch_type = {'EXG1': 'eog', 'EXG2': 'eog',  # vertical eye movements
           'EXG3': 'eog', 'EXG4': 'eog'}  # horizontal eye movements
ch_drop = ['EXG5', 'EXG6', 'EXG7', 'EXG8']  # unused external channels
ch_loc = 'biosemi64'  # biosemi montage

# segmentation and light filtering parameters
trigger = 71  # fixation cross trigger
n_blocks = 6 # number of recording blocks
n_freqs = [50, 100]  # notch filter frequencies
l_freq = 1.0  # high-pass filter cutoff
h_freq = 100.0  # low-pass filter cutoff

# preprocessed data storage
raw_dict = {}

################################################################

# pre-filtering loop for participants with 270 trials, evenly distributed across 6 blocks
# (main001, main002, main003, main004, main007, main008, main009, main010, main012)
for id in ID_list:
    if id in ID_excl:
        continue

    # set file name
    file_name = fname_ift % id

    # load data & set channel information
    raw = set_channels(data_path=dir_prepro, file_name=file_name, ch_type=ch_type, ch_drop=ch_drop, ch_loc=ch_loc)

    # apply light filters (to each block individually to avoid edge artifacts); returns continuous filtered EEG file
    raw_filt = segment_and_filter(raw=raw, cross_trigger=trigger, n_blocks=n_blocks, n_freqs=n_freqs, l_freq=l_freq, h_freq=h_freq)

    # store in dictionary & save to disk
    raw_dict[id] = raw_filt
    raw_filt.save(dir_prepro + fname_filt % id, overwrite=True)

################################################################

# pre-processing loop: interpolation, re-referencing, down-sampling, final filtering
for id in ID_list:
    if id in ID_excl:
        continue

    # select pre-filt loaded data / load them from disk
    #raw_filt = raw_dict[id]
    raw_filt = mne.io.read_raw_fif(dir_prepro + fname_filt % id, preload=True)

    # interpolate, re-referencing, down-sampling, final filters
    raw_prepro = preprocess(raw_filt)

    # save pre-processed data
    raw_dict[id] = raw_prepro
    raw_prepro.save(dir_prepro + fname_prepro % id, overwrite=True)





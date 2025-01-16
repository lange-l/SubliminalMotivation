"""
author: Lena Lange
last update: 16.01.2025
Python 3.13

Replication of Pessiglione et al. (2007) using unmasked stimuli
INCENTIVE FORCE TASK

Cut continuous raw data to only IFT main segment and save on disk

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

dir_main = 'C:/Users/Admin/Documents/PessRep_eeg_data/'
dir_prepro = 'C:/Users/Admin/Documents/PessRep_eeg_data/prepro/'

fname_raw = r'unmask_%s.bdf'
fname_ift = r'%s_IFT_raw.fif'
sfreq = 2048

#####################################################################################
### participants with 1 IFT_start and 1 IFT_end trigger (44 & 45)

IDs = ['main003', 'main007', 'main008', 'main009', 'main010']

for id in IDs:

    # load raw continuous data
    raw = mne.io.read_raw_bdf(dir_main + fname_raw % id, preload=True)
    events = mne.find_events(raw)

    # get start time
    start_event = events[events[:, 2] == 44] # [overall_sample, 0, trigger_id]
    start_sample = start_event[0, 0] # int
    start_time = start_sample / sfreq # float

    # get end time
    end_event = events[events[:, 2] == 45]
    end_sample = end_event[0, 0] # int
    end_time = end_sample / sfreq

    # crop
    raw_crop = raw.copy()
    raw_crop.crop(tmin = start_time, tmax = end_time)

    # get events of cropped segment
    events_crop = mne.find_events(raw_crop)
    raw_crop.plot(events=events, n_channels=64, duration=25)
    plt.show(block=True)

    # save
    raw_crop.save(dir_prepro + fname_ift % id)

#####################################################################################
### main002: IFT_end (45) trigger missing

# load raw continuous data
id = r'main002'
raw = mne.io.read_raw_bdf(dir_main + fname_raw % id, preload=True)
events = mne.find_events(raw)

# get start time
start_event = events[events[:, 2] == 44] # [overall_sample, 0, trigger_id]
start_sample = start_event[0, 0] # int
start_time = start_sample / raw.info['sfreq'] # float

# get time of last event from array (76 - rewardOFF)
end_event = events[-1,:]
end_sample = end_event[0] # int
end_time = end_sample / raw.info['sfreq']

# crop
raw_crop = raw.copy()
raw_crop.crop(tmin = start_time, tmax = end_time)

# get events of cropped segment
events_crop = mne.find_events(raw_crop)
raw_crop.plot(events=events, n_channels=64, duration=25)
plt.show(block=True)

# save
raw_crop.save(dir_prepro + fname_ift % id)

#####################################################################################
### main004: concatenate 004 and 004B IFT segments (3 blocks + 3 blocks)

id = 'main004'
raw004 = mne.io.read_raw_bdf(dir_main + fname_raw % id, preload=True)
events004 = mne.find_events(raw004)
trials004 = events004[events004[:,2] == 71] # 207 trials in total

# get start time, count trials after trigger 44
start_event004 = events004[events004[:, 2] == 44] # [overall_sample, 0, trigger_id]
start_sample004 = start_event004[0, 0] # int
start_time004 = start_sample004 / sfreq # float
trials004_IFT = events004[(events004[:,2] == 71) & (events004[:,0] > start_sample004)] # 135 trials after IFT_start

# get end time: IFT_end missing, cut after last event (81 - PAS_answer)
end_event004 = events004[-1,:]
end_sample004 = end_event004[0] # int
end_time004 = end_sample004 / sfreq + 1

# crop
raw004_crop = raw004.copy()
raw004_crop.crop(tmin=start_time004, tmax=end_time004)

# plot
#events004_crop = mne.find_events(raw004_crop)
#raw004_crop.plot(events=events004_crop, n_channels=64, duration=25)


id = 'main004B'
raw004B = mne.io.read_raw_bdf(dir_main + fname_raw % id, preload=True)
events004B = mne.find_events(raw004B)
trials004B = events004B[events004B[:,2] == 71] # 135 trials in total

# get start time: take first fix. cross (2nd event of the array) minus 1 s
start_sample004B = events004B[1, 0] # int
start_time004B = start_sample004B / sfreq - 1 # float

# get end time, count trials between start and end sample
end_event004B = events004B[events004B[:, 2] == 45]
end_sample004B = end_event004B[0, 0] # int
end_time004B = end_sample004B / sfreq

trials004B_IFT = events004B[(events004B[:,2] == 71) & (events004B[:,0] >= start_sample004B) & (events004B[:,0] < end_sample004B)] # 135 trials between start and end

# crop
raw004B_crop = raw004B.copy()
raw004B_crop.crop(tmin=start_time004B, tmax=end_time004B)

# plot
#events004B_crop = mne.find_events(raw004B_crop)
#raw004B_crop.plot(events=events004B_crop, n_channels=64, duration=25)


# concatenate, plot & save
raw_004_conc = mne.concatenate_raws([raw004_crop, raw004B_crop])

events_conc = mne.find_events(raw_004_conc)
trials_conc = events_conc[events_conc[:,2] == 71] # 270 trials in total, 135 first half, 135 second half
raw_004_conc.plot(events=events_conc, n_channels=64, duration=25)

id = 'main004'
raw_004_conc.save(dir_prepro + fname_ift % id)


# #####################################################################################
# ### main005: MATLAB script crashed after 180 IFT trials, completed 180 more --> cut out middle segment
#
# # load raw continuous data
# id = 'main005' # 2 x 180 trials in 1 recording
# raw = mne.io.read_raw_bdf(dir_main + fname_raw % id, preload=True)
# events = mne.find_events(raw)
#
# # get start times
# start_events = events[events[:, 2] == 44]
# #([[1077409,       0,      44],
# #  [1480429,       0,      44]])
# start_samples = start_events[:, 0] # [1077409, 1480429]
#
# # get end times
# end_event_1 = events[(events[:, 2] == 65536) & (events[:, 0] > start_samples[0])][0] # first trigger 65536 that occurs after the first 44 trigger
# end_sample_1 = end_event_1[0]
#
# end_event_2 = events[(events[:, 2] == 65536) & (events[:, 0] > start_samples[1])][0] # first trigger 65536 that occurs after the second 44 trigger
# end_sample_2 = end_event_2[0]
#
# start_time_1 = start_samples[0] / sfreq
# end_time_1 = end_sample_1/sfreq
#
# start_time_2 = start_samples[1] / sfreq
# end_time_2 = end_sample_2/sfreq
#
# # crop
# raw_1 = raw.copy()
# raw_1.crop(tmin=start_time_1, tmax=end_time_1)
#
# raw_2 = raw.copy()
# raw_2.crop(tmin=start_time_2, tmax=end_time_2)
#
# # concatenate
# raw_conc = mne.concatenate_raws([raw_1, raw_2])
# events_conc = mne.find_events(raw_conc)
# raw_conc.plot(events=events_conc, n_channels=64, duration=25)
#
# # save
# raw_conc.save(dir_prepro + fname_ift % id)


#####################################################################################
### main006: recording crashed after 2 IFT blocks (006A), behav. data for last 4 blocks (006B) lost

# load raw continuous data
id = 'main006A' # only first 2 blocks, for other 4 no behav. data
raw = mne.io.read_raw_bdf(dir_main + fname_raw % id, preload=True)
events = mne.find_events(raw)

# get start time
start_event = events[events[:, 2] == 44] # array([[574640,      0,     44]])
start_sample = start_event[0, 0] # np.int64(574640)
start_time = start_sample / sfreq # np.float64(280.5859375)

# get end time (last event: 81 PAS_answer)
end_sample = events[-1, 0] # int
end_time = end_sample / sfreq

# crop
raw_crop = raw.copy()
raw_crop.crop(tmin=start_time, tmax=end_time)

# save
id = 'main006'
raw_crop.save(dir_prepro + fname_ift % id)


#####################################################################################
### main012: 6th block too long, cut after 270 trials

# load raw continuous data
id = 'main012B'
raw = mne.io.read_raw_bdf(dir_main + fname_raw % id, preload=True)
events = mne.find_events(raw)

# get start time
start_event = events[events[:, 2] == 44] # array([[91320,     0,    44]])
start_sample = start_event[0, 0] # np.int64(91320)
start_time = start_sample / sfreq # np.float64(44.58984375)

# get end time: get trigger 76 (rewardOFF) of all events after IFT_start, choose the 270th
all_trials = events[(events[:, 2] == 71) & (events[:, 0] > start_sample)]
end_sample = all_trials[270, 0]  # get sample number
end_time = end_sample / sfreq - 0.5  # convert to time

# crop
raw_crop = raw.copy()
raw_crop.crop(tmin=start_time, tmax=end_time)

# get events of cropped segment
events_crop = mne.find_events(raw_crop)
print(f"Number of trials: {len(events_crop[events_crop[:, 2] == 71])}")

# save
id='main012'
raw_crop.save(dir_prepro + fname_ift % id)


#####################################################################################
### main014: task stopped working before the end (red filling bar error)

# load raw continuous data
id = 'main014'
raw = mne.io.read_raw_bdf(dir_main + fname_raw % id, preload=True)
events = mne.find_events(raw)

# get start time
start_event = events[events[:, 2] == 44]
start_sample = start_event[0, 0]
start_time = start_sample / sfreq

# get end time (last event: 75 rewardON)
end_sample = events[-1, 0] # int
end_time = end_sample / sfreq + 2

# crop
raw_crop = raw.copy()
raw_crop.crop(tmin=start_time, tmax=end_time)

# save
id = 'main014'
raw_crop.save(dir_prepro + fname_ift % id)

import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_model

data = pd.read_csv('data/gip_A00054817001.csv')
events = pd.read_csv('data/gip_A00054817001_events.csv')


# The first 20 mark is at latency 31225
data.columns[31225]

events['type'] = events['type'].astype('string')
event_lt = list(events['latency'].loc[events['type'] == '20  '].astype('int'))

eyes_open = []

for latency in event_lt:
    try:
        chunk = data.loc[: ,data.columns[latency]:data.columns[latency + 10000]]
    except:
        chunk = data.loc[: ,data.columns[latency]:]
    eyes_open.append(chunk)

i = 1
chunk_list = []
window_size_seconds = 2
window_size_miliseconds = window_size_seconds * 1000

# Going through each chunk in eyes open
for chunk in eyes_open:
    # Selecting each channel from the chunk
    for channel in range(len(chunk)):
        sub_ch = chunk.iloc[channel, :].to_numpy(dtype='float')
        # Feeding the single channel/row of the chunk through pwelch function to return f, pxx
        f, pxx = scipy.signal.welch(sub_ch, fs = 500, nperseg = window_size_miliseconds)
        chunk_list.append((i, f, pxx))
    i += 1


chunk_df = pd.DataFrame(chunk_list, columns=['chunk', 'f', 'pxx'])

average_pxx = []
for channel in range(len(chunk_df.loc[chunk_df['chunk'] == 1]['pxx'])):
    channel_avg_pxx = []
    for pxx_idx in range(len(chunk_df.loc[chunk_df['chunk'] == 1]['pxx'].iloc[0])):
        chunk1 = chunk_df.loc[chunk_df['chunk'] == 1]['pxx'].iloc[channel][pxx_idx]
        chunk2 = chunk_df.loc[chunk_df['chunk'] == 2]['pxx'].iloc[channel][pxx_idx]
        chunk3 = chunk_df.loc[chunk_df['chunk'] == 3]['pxx'].iloc[channel][pxx_idx]
        chunk4 = chunk_df.loc[chunk_df['chunk'] == 4]['pxx'].iloc[channel][pxx_idx]
        chunk5 = chunk_df.loc[chunk_df['chunk'] == 5]['pxx'].iloc[channel][pxx_idx]
        channel_total_per_timestep = chunk1 + chunk2 + chunk3 + chunk4 + chunk5
        average_per_channel_timestep = channel_total_per_timestep / 5
        
        channel_avg_pxx.append(average_per_channel_timestep)
    print(channel_avg_pxx)
    average_pxx.append(channel_avg_pxx)

fm1 = FOOOF(min_peak_height=0.05, verbose=False)
f = chunk_df.loc[chunk_df['chunk'] ==  1]['f'][0]
powers1 = np.array(average_pxx[0])
plot_spectrum(f, powers1, log_powers=True,
              color='black', label='Original Spectrum')
fm1.fit(f, powers1)
fm1.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'})

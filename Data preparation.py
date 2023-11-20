#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import pandas as pd
import numpy as np
import pickle

from IPython.display import Audio
import librosa
import tempfile
import subprocess
import soundfile as sf

from enum import Enum
import gc

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# In[2]:


HOP_SIZE = 256
N_FFT = 1024
SAMPLING_RATE = 22050


# In[3]:


data_path_orig = 'D:\\lpnu\\магістратура\\диплом\\musdb18'
data_path_wav = 'D:\\lpnu\\магістратура\\диплом\\musdb18_wav'


# In[4]:


# class syntax
class Channel(Enum):
    MIXTURE = 0
    DRUMS = 1
    BASS = 2
    OTHER = 3
    VOCALS = 4


# ## Stems to wav

# In[5]:


def to_wav(path, sr):
    with tempfile.TemporaryDirectory() as tmpdir:
        for subdir in ('train', 'test'):
            origin_dir = os.path.join(path, subdir)
            files = [f for f in os.listdir(origin_dir)
                     if os.path.splitext(f)[1] == '.mp4']
            for file in files:
                path = os.path.join(origin_dir, file)
                name = os.path.splitext(file)[0]
                wav_data = []
                # Extract & save the sound of `ch` channel to a temp directory
                # and then concatenate all channels to a single .wav file
                for ch in range(5):
                    temp_fn = f'{name}.{ch}.wav'
                    out_path = os.path.join(tmpdir, temp_fn)
                    subprocess.run(['ffmpeg', '-i', path,
                                    '-map', f'0:{ch}', out_path])
                    sound, _ = librosa.load(out_path, sr=sr, mono=True)
                    wav_data.append(sound)
                wav_data = np.stack(wav_data, axis=1)
                out_path = os.path.join(
                    data_path_wav, subdir, f'{name}.wav')
                sf.write(out_path, wav_data, sr)


# In[6]:


def load_df(path, subdir, offset=30, duration=60):
    # Initialize lists to store information
    file_paths = []
    file_names = []
    music_length = []
    sample_rates = []
    music_variations = []

    # Define the directory to search for files
    origin_dir = os.path.join(path, subdir)

    # Iterate through files in the directory and its subdirectories
    for root, _, files in os.walk(origin_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                file_names.append(file)

                # Read the WAV file using wavfile.read
                song_duration = librosa.get_duration(path=file_path)
                start = 0
                if offset < song_duration/3:
                    start = offset
                    
                audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False, offset=start, duration=duration)
                    
                sample_rates.append(sample_rate)
                
                music_length.append(song_duration)
                #print(type(audio_data[0])
                
                # Append the 5 music variations to the list
                music_variations.append(audio_data.T.astype(np.float32))
                
                del file_path, audio_data, sample_rate
                gc.collect()

    # Create a DataFrame from the collected information
    df = pd.DataFrame({
        'File_Path': file_paths,
        'File_Name': file_names,
        'Music_Length': music_length,
        'Sample_Rate': sample_rates,
    })

    # Add columns for each music variation
    for i in range(5):
        """Output .wav files contain 5 channels
        - `0` - The mixture,
        - `1` - The drums,
        - `2` - The bass,
        - `3` - The rest of the accompaniment,
        - `4` - The vocals.
        """
        df[Channel(i).name] = [variation[:, i] for variation in music_variations]
    
    del file_paths, file_names,sample_rates, music_variations
    gc.collect()
    return df


# In[12]:


df = load_df(data_path_wav, 'test', )
df.head()


# In[13]:


df.to_pickle('dataset_test.pkl')


# In[6]:


df = pd.read_pickle('dataset_test.pkl')


# In[9]:


df.head()['MIXTURE'][2].shape


# ## WAV to STFT

# In[11]:


df = pd.read_pickle('dataset_test.pkl')
df.head()


# In[13]:


df['File_Name'][26]


# In[14]:


df['MIXTURE'][26]


# In[68]:


display(Audio(data=df['MIXTURE'][0], rate=SAMPLING_RATE)) 


# In[7]:


def get_stft(audio_data, sr=22050, n_fft=1024, hop_length=256):
    spec = librosa.stft(audio_data, hop_length=hop_length, n_fft=n_fft )
    db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    gc.collect()
    return spec.astype(np.float32), db.astype(np.float32)


# In[8]:


def plot_stft(audio_data, sr=22050, n_fft=1024, hop_length=256, cmap='viridis'):
    plt.figure(figsize=(12, 8))
    
    spec, db = get_stft(audio_data, sr, n_fft, hop_length)
    #print(len(audio_data))
    #print(db.T.shape)
    librosa.display.specshow(db, sr=sr, hop_length=hop_length, n_fft=n_fft, y_axis='log', x_axis='time')
    #display(Audio(data=librosa.istft(spec.astype(np.complex64), hop_length=hop_length, n_fft=win_length), rate=sr))
    
    #spec = np.abs(librosa.stft(audio_data, hop_length=hop_length, win_length=win_length))
    #librosa.display.specshow(spec, sr=sr, hop_length=hop_length, win_length=win_length, y_axis='log', x_axis='time')
    plt.show()


# In[9]:


def create_stft_df(df, sr=22050, n_fft=1024, hop_length=256, threshold=1):
    df_new = pd.DataFrame(index=range(len(df)))
    for index in range(5):
        df_new[Channel(index).name+'_STFT_SPEC'] = None
        if index != 0:
            df_new[Channel(index).name+'_STFT_MASK'] = None
        print(Channel(index).name)
        for j, song in df[Channel(index).name].items():
            spec, _ = get_stft(song, sr, n_fft, hop_length)
            df_new.at[j, Channel(index).name+'_STFT_SPEC'] = spec
            if index != 0:
                mixture_magnitude = df_new['MIXTURE_STFT_SPEC'][j]
                mask = (np.abs(spec) > threshold * np.abs(mixture_magnitude))
                df_new.at[j, Channel(index).name+'_STFT_MASK'] = mask
            gc.collect()
    
    return df_new


# In[72]:


df_stft = create_stft_df(df, sr=22050, n_fft=1024, hop_length=256, threshold=0.6)
df_stft.head()


# In[46]:


df_stft.to_pickle('stft_dataset_test.pkl')


# ## Preprocess STFT dataframe for trining

# In[10]:


def sliding_window_middle(lst, n=25):
    result = []
    lst_len = len(lst)
    for i in range(lst_len):
        window = []
        for j in range(n):
            idx = i - n // 2 + j
            if idx < 0:
                window.append(lst[0])
            elif idx >= lst_len:
                window.append(lst[-1])
            else:
                window.append(lst[idx])
        result.append(window)

    return np.array(result)


# In[11]:


def process_stft_df(df, n=25):
    new_df = df
    n_mels=513
    sr=22050
    n_fft=1024
    mel_filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    for column in df.columns:
        if column == 'MIXTURE_STFT_SPEC':
            print('special ' + column)
            #df_mel_col = df[column].apply( lambda row: np.transpose(sliding_window_middle(mel_filterbank.dot(row).T, n=25), (0, 2, 1)) )
            #new_df.insert(loc = 0, column = 'MIXTURE_STFT_MEL_FRAME', value = df_mel_col)
            
            df_mel_col = df[column].apply( lambda row: np.transpose(sliding_window_middle(row.T, n=25), (0, 2, 1)) )
            new_df.insert(loc = 0, column = 'MIXTURE_STFT_SPEC_FRAME', value = df_mel_col)
            
        print('general ' + column)
        new_df[column] = df[column].apply( lambda row: row.T )
        
    return new_df
        


# In[5]:


df = pd.read_pickle('stft_dataset_test.pkl')
df.head()


# In[6]:


df['MIXTURE_STFT_SPEC'][26]


# In[8]:


df['MIXTURE_STFT_SPEC'][26]


# In[7]:


df['MIXTURE_STFT_SPEC'][0].shape


# In[78]:


plt.figure(figsize=(12, 8))
librosa.display.specshow(df_stft['VOCALS_STFT_MASK'][4], sr=22050, hop_length=256, win_length=1024, y_axis='log', x_axis='time', cmap='magma')
plt.show()


# In[24]:


processed_df = process_stft_df(df)
processed_df.head()


# In[25]:


processed_df.to_pickle('processed_stft_test.pkl')


# In[26]:


processed_df.explode(list(processed_df.columns))


# In[27]:


processed_df.explode(list(processed_df.columns)).reset_index().rename(columns = {'index':'song'}).to_pickle('test_stft_nomel.pkl')


# In[28]:


gc.collect()


# ## To files/CSV

# In[5]:


processed_df=pd.read_pickle('test_stft.pkl')
processed_df.head()


# In[13]:


def make_usable(df, path):
    csv_df = pd.DataFrame(index=range(len(df)))
    csv_df['song'] = None
    csv_df['path'] = None
    for index, row in df.iterrows():
        file_path = path+'/frames/'+str(row.name)+'_'+str(row.song)+'_frame.pkl'
        row.to_pickle(file_path)
        csv_df.at[index, 'song'] = row.song
        csv_df.at[index, 'path'] = file_path
        
    csv_df.reset_index(inplace=True)
    csv_df.to_csv(path+'/frames.csv', index_label=False)
    return csv_df


# In[14]:


make_usable(processed_df, '../musdb18_data/test')


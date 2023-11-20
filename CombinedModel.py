import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from enum import Enum
import pandas as pd
import numpy as np
import random
import librosa
import sounddevice as sd

random.seed(47)
np.random.seed(47)
torch.manual_seed(47)

class Channel(Enum):
    MIXTURE = 0
    DRUMS = 1
    BASS = 2
    OTHER = 3
    VOCALS = 4

HOP_SIZE = 256
WINDOW_SIZE = 1024
SAMPLING_RATE = 22050

class CustomRegressionDataset(Dataset):
    def __init__(self, data):
        self.X = data['audio_data_frame']
        self.Mix = data['audio_data']

    def __len__(self):
        torch.cuda.empty_cache()
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        mix = self.Mix[idx]
        return torch.FloatTensor(x), torch.FloatTensor(mix)

class CustomRegressionCNN(nn.Module):
    def __init__(self):
        super(CustomRegressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding='same')
        self.relu2 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(16, 64, kernel_size=(3, 3), padding='same')
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, 16, kernel_size=(3, 3), padding='same')
        self.relu4 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout2 = nn.Dropout(0.1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 57 * 2, 128)
        self.relu5 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 513)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class Combined_model(nn.Module):
    def __init__(self, modelDrums, modelBass, modelOther, modelVocal):
        super(Combined_model, self).__init__()
        self.modelDrums = modelDrums
        self.modelBass = modelBass
        self.modelOther = modelOther
        self.modelVocal = modelVocal

    def forward(self, x):
        x1 = self.modelDrums(x)
        x2 = self.modelBass(x)
        x3 = self.modelOther(x)
        x4 = self.modelVocal(x)
        x = torch.cat((x1.detach(), x2.detach(), x3.detach(), x4.detach()), dim=1)
        return x

def load_ckp(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

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

def create_combineModel(drums = None, bass = None, other = None, vocal = None):
    drums = CustomRegressionCNN()
    drums = load_ckp('D:/lpnu/checkpoints/regression/drums/66_checkpoint.pt', drums)

    bass = CustomRegressionCNN()
    bass = load_ckp('D:/lpnu/checkpoints/regression/bass/50_checkpoint.pt', bass)

    other = CustomRegressionCNN()
    other = load_ckp('D:/lpnu/checkpoints/regression/other/52_checkpoint.pt', other)

    vocal = CustomRegressionCNN()
    vocal = load_ckp('D:/lpnu/checkpoints/regression/vocal/48_checkpoint.pt', vocal)

    model = Combined_model(drums, bass, other, vocal)

    return model

def music_to_dataloader(path = None):
    audio_data, sample_rate = librosa.load(path, sr=SAMPLING_RATE, mono=False, offset=30, duration=60)
    print(audio_data.shape)
    if len(audio_data.shape) == 5:
        audio_data = audio_data[0]
    elif len(audio_data.shape) == 2:
        audio_data=librosa.to_mono(audio_data)
    audio_data = librosa.stft(audio_data, hop_length=HOP_SIZE, n_fft=WINDOW_SIZE ).astype(np.float32)
    X = pd.DataFrame({'audio_data':pd.Series([audio_data])})
    X['audio_data_frame'] = X['audio_data'].apply( lambda row: np.transpose(sliding_window_middle(row.T, n=25), (0, 2, 1)) )
    X['audio_data'] = X['audio_data'].apply( lambda row: row.T )

    X = X.explode(list(X.columns)).reset_index()

    test_dataset = CustomRegressionDataset(X)
    dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    return dataloader

def predict(model, data_loader):
    result = torch.Tensor()
    mixture = torch.Tensor()
    with torch.no_grad():
        model.eval()
        with tqdm(data_loader) as bar:
            bar.set_description(f"Separation")
            for batch in bar:
                # take a batch
                inputs, mix = batch
                # forward pass
                outputs = model(inputs)
                result = torch.cat((result, outputs.round()), dim=0)
                mixture = torch.cat((mixture, mix), dim=0)

    # plt.figure(figsize=(12, 8))
    # librosa.display.specshow(result[:, 513 * 3:513 * 4].numpy().T.astype(np.complex64), sr=22050, hop_length=256, win_length=1024, y_axis='log', x_axis='time', cmap='magma')
    # plt.show()

    # sd.play(data=librosa.istft(mixture.numpy().T.astype(np.complex64) * result[:, 513 * 3:513 * 4].numpy().T.astype(np.complex64), hop_length=HOP_SIZE, n_fft=WINDOW_SIZE), samplerate=SAMPLING_RATE)
    # sd.wait()

    return mixture.numpy(), result.numpy()

# vocal = CustomRegressionCNN()
# vocal = load_ckp('D:/lpnu/checkpoints/regression/vocal/48_checkpoint.pt', vocal)
#
# combined_model = create_combineModel()
#test_dataloader = music_to_dataloader()
# predict(combined_model, test_dataloader)




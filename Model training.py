#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from tqdm import tqdm
from enum import Enum
import math
import numpy as np

from sklearn.model_selection import train_test_split


# In[2]:


torch.cuda.empty_cache()


# In[3]:


torch.__version__


# In[4]:


# class syntax
class Channel(Enum):
    MIXTURE = 0
    DRUMS = 1
    BASS = 2
    OTHER = 3
    VOCALS = 4


# In[5]:


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:


num_epochs = 100


# In[7]:


#device = torch.device('cpu')


# In[8]:


#torch.cuda.set_per_process_memory_fraction(0.5)


# In[9]:


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)
    print('Model saved!')


# In[10]:


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


# In[11]:


def save_history(train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist, path):
    df = pd.DataFrame({'train_acc_hist': train_acc_hist,
                       'train_loss_hist': train_loss_hist,
                       'valid_acc_hist': valid_acc_hist,
                       'valid_loss_hist': valid_loss_hist})
    df.to_csv(path+'history.csv', index_label=False)
    torch.cuda.empty_cache()


# In[12]:


def plot_history(train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist, path):
    plt.plot(range(len(train_acc_hist)), train_acc_hist, "-b", label="train_acc")
    plt.plot(range(len(valid_acc_hist)), valid_acc_hist, "-r", label="valid_acc")
    plt.title('Accuracy')
    plt.legend(loc="upper left")
    plt.savefig(path+'accuracy.png', bbox_inches='tight')
    plt.show(valid_acc_hist)
    torch.cuda.empty_cache()

    plt.plot(range(len(train_loss_hist)), train_loss_hist, "-b", label="train_loss")
    plt.plot(range(len(valid_loss_hist)), valid_loss_hist, "-r", label="valid_loss")
    plt.title('Loss')
    plt.legend(loc="upper left")
    plt.savefig(path+'loss.png', bbox_inches='tight')
    plt.show()
    torch.cuda.empty_cache()


# In[13]:


def get_history(path):
    try:
        df = pd.read_csv(path)
        if not df.empty:
            return df['train_acc_hist'].to_numpy().tolist(), df['train_loss_hist'].to_numpy().tolist(), df['valid_acc_hist'].to_numpy().tolist(), df['valid_loss_hist'].to_numpy().tolist()
        else:
            return [0.0], [1.0], [0.0], [1.0]
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        return [0.0], [1.0], [0.0], [1.0]


# ## Sound detection

# In[14]:


class CustomClassificationDataset(Dataset):
    def __init__(self, data, X_column, Y_column):
        self.X_column = X_column
        self.Y_column = Y_column
        self.data = data
        
    def __len__(self):
        torch.cuda.empty_cache()
        return len(self.data)
    
    def __getitem__(self, idx):
        frame = pd.read_pickle(self.data.iloc[idx]['path'])
        X = torch.FloatTensor(frame[self.X_column])
        Y = torch.FloatTensor(frame[self.Y_column])   
        
        X_dev = X.to(device)
        Y_dev = (torch.sum(Y) > 30).to(device)
        torch.cuda.empty_cache()
        return X_dev, Y_dev


# In[15]:


data = pd.read_csv('../musdb18_data/train/frames.csv')
data.head()


# In[16]:


test_data = pd.read_csv('../musdb18_data/test/frames.csv')
test_data.head()


# In[17]:


data.info(memory_usage = "deep")


# In[18]:


test, validation = train_test_split(test_data, test_size=0.4, random_state=47)


# In[19]:


train = data.sample(frac=1, random_state=47).reset_index(drop=True)


# In[20]:


test.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)


# In[21]:


test


# In[22]:


train


# In[23]:


validation


# In[24]:


# Create an instance of your custom dataset
train_dataset = CustomClassificationDataset(train, 'MIXTURE_STFT_SPEC_FRAME', 'VOCALS_STFT_MASK')
val_dataset = CustomClassificationDataset(validation, 'MIXTURE_STFT_SPEC_FRAME', 'VOCALS_STFT_MASK')
test_dataset = CustomClassificationDataset(test, 'MIXTURE_STFT_SPEC_FRAME', 'VOCALS_STFT_MASK')


# In[25]:


train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=False
)

val_dataloader = DataLoader(
    val_dataset, batch_size=128, shuffle=False
)

test_dataloader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)
#5168


# In[26]:


class CustomClassificationCNN(nn.Module):
    def __init__(self):
        super(CustomClassificationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding='same')
        self.relu2 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(16, 64, kernel_size=(3, 3), padding='same')
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, 16, kernel_size=(3, 3), padding='same')
        self.relu4 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout2 = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 57 * 2, 128)
        self.relu5 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 1)
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


# In[25]:


# Instantiate the model
model = CustomClassificationCNN()
summary(model, input_size=(None, 513, 25, 16))


# In[26]:


model.to(device)


# In[33]:


# Define the loss and optimizer
criterion_class = nn.BCELoss()
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.99, nesterov=True)


# In[34]:


criterion = criterion_class
criterion.to(device)
criterion


# In[39]:


optimizer = optimizer_sgd
optimizer


# In[36]:


model, optimizer, start_epoch = load_ckp('checkpoints/classification/16_checkpoint.pt', model, optimizer)


# In[37]:


model.to(device)


# In[38]:


optimizer


# In[33]:


scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                     base_lr = 0.005, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                                     max_lr = 0.05, # Upper learning rate boundaries in the cycle for each parameter group
                                     base_momentum = 0.9,
                                     max_momentum = 0.99,
                                     step_size_up = 4, # Number of training iterations in the increasing half of a cycle
                                     last_epoch = start_epoch-2,
                                     mode = "triangular")


# In[40]:


train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist = get_history('checkpoints/classification/history.csv')


# In[41]:


train_loss_hist


# In[42]:


for epoch in range(start_epoch, num_epochs+1):
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    running_val_loss = 0.0

    predicted_labels = torch.Tensor().to(device)
    true_labels = torch.Tensor().to(device)

    with tqdm(train_dataloader) as bar:
        bar.set_description(f"Epoch {epoch}")
        for batch in bar:
            # take a batch
            torch.cuda.empty_cache()
            inputs, labels = batch 
            inputs, labels = inputs.to(device), labels.float().to(device)

            # forward pass
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            torch.cuda.empty_cache()
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # print progress
            running_train_loss += loss.item()
            acc = (outputs.round() == labels).float().mean()
            
            predicted_labels = torch.cat((predicted_labels, outputs.round()), dim=0)
            true_labels = torch.cat((true_labels, labels), dim=0)
            torch.cuda.empty_cache()

            bar.set_postfix(
                loss=float(loss),
                acc=float(acc),
                running_train_loss=float(running_train_loss / (bar.n + 1))
            )
    
    final_acc = (predicted_labels == true_labels).float().mean()
    final_loss = running_train_loss/len(train_dataloader)
    print('accuracy: ', final_acc.item())
    print('loss: ', final_loss)
    train_acc_hist.append(final_acc.item())
    train_loss_hist.append(final_loss)

    del predicted_labels, true_labels
    torch.cuda.empty_cache()

    predicted_labels = torch.Tensor().to(device)
    true_labels = torch.Tensor().to(device)

    torch.cuda.empty_cache()

    with torch.no_grad():
        model.eval()
        with tqdm(val_dataloader) as bar:
            bar.set_description(f"Validation")
            for batch in bar:
                # take a batch
                torch.cuda.empty_cache()
                inputs, labels = batch 
                inputs, labels = inputs.to(device), labels.float().to(device)

                # forward pass
                outputs = model(inputs).squeeze(1)
                val_loss = criterion(outputs, labels)
                torch.cuda.empty_cache()
                
                # print progress
                running_val_loss += val_loss.item()
                val_acc = (outputs.round() == labels).float().mean()

                predicted_labels = torch.cat((predicted_labels, outputs.round()), dim=0)
                true_labels = torch.cat((true_labels, labels), dim=0)
                torch.cuda.empty_cache()

                bar.set_postfix(
                    loss=float(val_loss),
                    acc=float(val_acc),
                )
                
    final_acc = (predicted_labels == true_labels).float().mean()
    final_loss = running_val_loss/len(val_dataloader)
    print('accuracy: ', final_acc.item())
    print('loss: ', final_loss)
    valid_acc_hist.append(final_acc.item())
    valid_loss_hist.append(final_loss)

    del predicted_labels, true_labels
    torch.cuda.empty_cache()

    #scheduler.step()

    if epoch%2==0:
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, 'checkpoints/classification/'+str(epoch)+'_checkpoint.pt')
        
        torch.cuda.empty_cache()
        save_history(train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist, 'checkpoints/classification/')
        torch.cuda.empty_cache()
        plot_history(train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist, 'checkpoints/classification/')
        torch.cuda.empty_cache()
        
    torch.cuda.empty_cache()

print("Training complete")


# ## Sound separation

# In[14]:


class CustomRegressionDataset(Dataset):
    def __init__(self, data, X_column, Y_column):
        self.X_column = X_column
        self.Y_column = Y_column
        self.data = data
        #self.scaler = StandardScaler()
        
    def __len__(self):
        torch.cuda.empty_cache()
        return len(self.data)
    
    def __getitem__(self, idx):
        frame = pd.read_pickle(self.data.iloc[idx]['path'])
        X = frame[self.X_column]
        Y = frame[self.Y_column]
        
        X_dev = torch.FloatTensor(X).to(device)
        Y_dev = torch.FloatTensor(Y).to(device)
            
        torch.cuda.empty_cache()
        return X_dev, Y_dev


# In[42]:


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


# In[43]:


class CustomRegressionCNN_Norm(nn.Module):
    def __init__(self):
        super(CustomRegressionCNN_Norm, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after the first convolution
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(16)  # Batch normalization after the second convolution
        self.relu2 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2d(16, 64, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization after the third convolution
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, 16, kernel_size=(3, 3), padding='same')
        self.bn4 = nn.BatchNorm2d(16)  # Batch normalization after the fourth convolution
        self.relu4 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout2 = nn.Dropout(0.1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 57 * 2, 128)
        self.bn5 = nn.BatchNorm1d(128)  # Batch normalization before the first fully connected layer
        self.relu5 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(128, 513)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)  # Batch normalization
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)  # Batch normalization
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.bn3(x)  # Batch normalization
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)  # Batch normalization
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn5(x)  # Batch normalization
        x = self.relu5(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# In[44]:


class CustomRegressionCNN_Imp(nn.Module):
    def __init__(self):
        super(CustomRegressionCNN_Imp, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization after the first convolution
        self.relu1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization after the first convolution
        self.relu2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(16)  # Batch normalization after the second convolution
        self.relu3 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same')
        self.bn4 = nn.BatchNorm2d(32)  # Batch normalization after the third convolution
        self.relu4 = nn.LeakyReLU()
        
        self.conv5 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.bn5 = nn.BatchNorm2d(64)  # Batch normalization after the third convolution
        self.relu5 = nn.LeakyReLU()
        
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding='same')
        self.bn6 = nn.BatchNorm2d(32)  # Batch normalization after the fourth convolution
        self.relu6 = nn.LeakyReLU()
        
        self.conv7 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding='same')
        self.bn7 = nn.BatchNorm2d(16)  # Batch normalization after the fourth convolution
        self.relu7 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout2 = nn.Dropout(0.1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 57 * 2, 128)
        self.bn8 = nn.BatchNorm1d(128)  # Batch normalization before the first fully connected layer
        self.relu8 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(128, 513)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)  # Batch normalization
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)  # Batch normalization
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)  # Batch normalization
        x = self.relu3(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv4(x)
        x = self.bn4(x)  # Batch normalization
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)  # Batch normalization
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)  # Batch normalization
        x = self.relu6(x)
        
        x = self.conv7(x)
        x = self.bn7(x)  # Batch normalization
        x = self.relu7(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn8(x)  # Batch normalization
        x = self.relu8(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# In[45]:


data = pd.read_csv('../musdb18_data/train/frames.csv')
data.head()


# In[46]:


test_data = pd.read_csv('../musdb18_data/test/frames.csv')
test_data.head()


# In[47]:


data.info(memory_usage = "deep")


# In[48]:


test, validation = train_test_split(test_data, test_size=0.4, random_state=47)


# In[49]:


train = data.sample(frac=1, random_state=47).reset_index(drop=True)


# In[50]:


test.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)


# In[51]:


train


# In[52]:


test


# In[53]:


validation


# In[54]:


target = Channel(2).name + "_STFT_MASK"
target


# In[55]:


# Create an instance of your custom dataset
train_dataset = CustomRegressionDataset(train, 'MIXTURE_STFT_SPEC_FRAME', target)
val_dataset = CustomRegressionDataset(validation, 'MIXTURE_STFT_SPEC_FRAME', target)
test_dataset = CustomRegressionDataset(test, 'MIXTURE_STFT_SPEC_FRAME', target)


# In[56]:


train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=False
)

val_dataloader = DataLoader(
    val_dataset, batch_size=16, shuffle=False
)

test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False
)
#5168


# In[57]:


# Instantiate the model
model = CustomRegressionCNN()
model.to(device)
summary(model, input_size=(16, 1, 513, 125))


# In[58]:


criterion_regress = nn.MSELoss()
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.99, nesterov=True)


# In[59]:


criterion = criterion_regress
criterion.to(device)
criterion


# In[60]:


optimizer = optimizer_sgd
optimizer


# In[61]:


model, optimizer, start_epoch = load_ckp('checkpoints/regression/bass/38_checkpoint.pt', model, optimizer)


# In[62]:


start_epoch


# In[63]:


scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                     base_lr = 0.01, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                                     max_lr = 0.05, # Upper learning rate boundaries in the cycle for each parameter group
                                     base_momentum = 0.9,
                                     max_momentum = 0.99,
                                     step_size_up = 4, # Number of training iterations in the increasing half of a cycle
                                     last_epoch = start_epoch-2,
                                     mode = "triangular")


# In[64]:


train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist = get_history('checkpoints/regression/bass/history.csv')


# In[65]:


train_acc_hist


# In[66]:


train_loss_hist


# In[67]:


valid_acc_hist


# In[68]:


for epoch in range(start_epoch, num_epochs+1):
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    running_val_loss = 0.0
    
    running_train_accuracy = 0.0
    running_val_accuracy = 0.0
    
    predicted_labels = torch.Tensor().to(device)
    true_labels = torch.Tensor().to(device)

    with tqdm(train_dataloader) as bar:
        bar.set_description(f"Epoch {epoch}")
        for batch in bar:
            # take a batch
            inputs, labels = batch 
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # print progress
            running_train_loss += loss.item()
            acc = (outputs.round() == labels).float().mean()

            predicted_labels = torch.cat((predicted_labels, outputs), dim=0)
            true_labels = torch.cat((true_labels, labels), dim=0)

            running_train_accuracy = (predicted_labels.round() == true_labels).float().mean().cpu().item()

            torch.cuda.empty_cache()
            #print(predicted_labels)

            bar.set_postfix(
                loss=float(loss),
                acc=float(acc),
                running_train_accuracy=float(running_train_accuracy),
                running_train_loss=float(running_train_loss / (bar.n + 1))
            )

    train_acc_hist.append(running_train_accuracy)
    train_loss_hist.append(running_train_loss/len(train_dataloader))

    del predicted_labels, true_labels
    torch.cuda.empty_cache()

    predicted_labels = torch.Tensor().to(device)
    true_labels = torch.Tensor().to(device)

    torch.cuda.empty_cache()

    with torch.no_grad():
        model.eval()
        with tqdm(val_dataloader) as bar:
            bar.set_description(f"Validation")
            for batch in bar:
                # take a batch
                inputs, labels = batch 
                inputs, labels = inputs.to(device), labels.to(device)  

                # forward pass
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                # print progress
                running_val_loss += val_loss.item()
                val_acc = (outputs.round() == labels).float().mean()

                predicted_labels = torch.cat((predicted_labels, outputs), dim=0)
                true_labels = torch.cat((true_labels, labels), dim=0)

                running_val_accuracy = (predicted_labels.round() == true_labels).float().mean().cpu().item()

                torch.cuda.empty_cache()

                bar.set_postfix(
                    loss=float(val_loss),
                    acc=float(val_acc),
                    running_val_accuracy=float(running_val_accuracy),
                    running_val_loss=float(running_val_loss / (bar.n + 1))
                )

    valid_acc_hist.append(running_val_accuracy)
    valid_loss_hist.append(running_val_loss/len(val_dataloader))

    del predicted_labels, true_labels
    torch.cuda.empty_cache()
    
    #test_loop(test_data, 'MIXTURE_STFT_SPEC_FRAME', 'VOCALS_STFT_MASK')
    scheduler.step()

    if epoch%2==0:
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, 'checkpoints/regression/bass/'+str(epoch)+'_checkpoint.pt')
        torch.cuda.empty_cache()
        save_history(train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist, 'checkpoints/regression/bass/')
        torch.cuda.empty_cache()
        plot_history(train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist, 'checkpoints/regression/bass/')
        torch.cuda.empty_cache()

print("Training complete")


# In[52]:


torch.cuda.empty_cache()


# In[43]:


torch.save(model.state_dict(), 'complete_model.pth')


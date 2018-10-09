import torch
import torch.nn as nn
import AudioDataset as dsets
from torch.utils.data.sampler import SubsetRandomSampler
import data_loader
import cPickle
import numpy as np

# Hyper Parameters
input_size = 39 # MFCC coeff 13 * 40
hidden_size = 500 # Not used
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

hidden_size1 = 100
hidden_size2 = 80
#hidden_size3 = 50

# CUP Specified
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use default data loader

train_dataset = dsets.AudioDataset('/Users/FuYiran/Desktop/EE 443/final_project/audio_data/1900newdata.txt', '/Users/FuYiran/Desktop/EE 443/final_project/audio_data/1900newlabel.txt')
test_dataset = dsets.AudioDataset('/Users/FuYiran/Desktop/EE 443/final_project/audio_data/data.txt', '/Users/FuYiran/Desktop/EE 443/final_project/audio_data/label.txt')

# Data Loader (Input Pipeline)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# Use train/validation split data loader

#train_loader, valid_loader = data_loader.get_train_valid_loader(data_dir='/Users/FuYiran/Desktop/EE 443/final_project/audio_data/', batch_size = batch_size, valid_size=0.01)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model (1 hidden layer)

'''class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out'''

# modify to create 2 hidden layers network

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        #self.fc2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        #self.fc3 = nn.Linear(hidden_size2, hidden_size3, bias=False)
        self.fc2 = nn.Linear(hidden_size1, num_classes, bias=False)
        self.float()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        #out = self.fc4(out)
        return out

# 1 Hidden Layer net
#net = Net(input_size, hidden_size, num_classes)

net = Net(input_size, hidden_size1, num_classes)
#net = nn.float()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#print(net)

# Train the Model
for epoch in range(num_epochs):
    for i, (audio, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable

        audio = audio.reshape(-1, 39).to(device)
        labels = labels.to(device)
        #print(labels)
        #break

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(audio)
        #print(outputs)
        #print(labels)
        #break
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

# Validate the Model
'''
correct_valid = 0
total_valid = 0
for audio, labels in valid_loader:
    audio = audio.reshape(-1, 39).to(device)
    outputs = net(audio)
    _, predicted = torch.max(outputs.data, 1)
    #print(predicted)
    total_valid += labels.size(0)
    correct_valid += (predicted == labels).sum()

print('Accuracy of the network on the validation audio: %d %%' % (100 * correct_valid / total_valid))
'''

# Test the Model
correct_test = 0
total_test = 0
for audio, labels in test_loader:
    audio = audio.reshape(-1, 39).to(device)
    outputs = net(audio)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    total_test += labels.size(0)
    correct_test += (predicted == labels).sum()

print('Accuracy of the network on the test audio: %d %%' % (100 * correct_test / total_test))

# Save the Model
torch.save(net.state_dict(), 'audio_model.pkl')
#print(net.state_dict())

'''
the_model = Net(input_size, hidden_size1, num_classes)
the_model.load_state_dict(torch.load("1_100_model.pkl"))
correct_test = 0
total_test = 0
for audio, labels in test_loader:
    audio = audio.reshape(-1, 39).to(device)
    outputs = the_model(audio)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    total_test += labels.size(0)
    correct_test += (predicted == labels).sum()

print('Accuracy of the network on the test audio: %d %%' % (100 * correct_test / total_test))
'''

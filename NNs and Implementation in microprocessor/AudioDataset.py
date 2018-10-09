import os
import numpy as np
import torch
import torch.utils.data as data

class AudioDataset(data.Dataset):

    def __init__(self, path_file, label_file):
        self.dataset = []
        self.label = []
        file1 = open(path_file)
        mfccs = file1.readlines()
        for line in mfccs:
            mfcc_array = line.split(' ')


            mfcc_array.remove('\n')

            input = np.asarray(mfcc_array, dtype=np.float32)

            self.dataset.append(input)
        file2 = open(label_file)
        labels = file2.readlines()
        #print(self.dataset)

        for line in labels:
            label_array = line.split('\n')
            #label_array.remove('')
            self.label.append(label_array[0])

        self.label = np.asarray(self.label, dtype=np.int_)
        #print self.label
        self.rows = len(self.dataset)



    def __getitem__(self, idx):
        return self.dataset[idx], self.label[idx]

    def __len__(self):
        return self.rows

if __name__ == '__main__':
    dset = AudioDataset('/Users/FuYiran/Desktop/EE 443/final_project/audio_data/testAudio.txt', '/Users/FuYiran/Desktop/EE 443/final_project/audio_data/testLabel.txt')
    dloader = data.DataLoader(dset)
    #for i, (img, label) in enumerate(dloader):
        #print img.size()
        #print label.size()
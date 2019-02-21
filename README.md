# MFCC-Processed-Digit-Recognition-System
This is the Capstone Project for Electrical Engineering DSP concentration at University of Washington.

CAPTIONS:

Neural Networks; Machine Learning; MFCC; Speech Recognition; Audio Classification;

DETAILS:

It is a real time audio digit number recognition system. It contains 2500 audio data for training and testing, 
1500 from public dataset, and 100 from volunteers within college. To deal with the different volume and length of input audio, I used the Mel-Frequency Cepstrum Coefficients(MFCC) 
to extract the features of input audio, which successfully reduce the data dimension to 34 by 1, and at the same time kept
the frequency features for network training. Finally, I implemented the feedforward neural nets for the training part and
fine tuned it for a good results.

STRUCTURE:

This Repo mainly contains two parts of code:
1. Matlab code is used to calculate MFCC of audio signals.
2. Python code is the implementation of Feedforward Neural Networks.

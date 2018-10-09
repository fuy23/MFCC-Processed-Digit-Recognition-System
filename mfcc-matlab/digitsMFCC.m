% Capstone MFCC Calculation
clear all; close all; clc

A = 25e-3; B = 10e-3;
alpha = 0.97;      % preemphasis coefficient
R = [ 300 3700 ];  % frequency range to consider
M = 20;            % number of filterbank channels 
C = 13;            % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter

numAudio = 10;
folder = '/Users/FuYiran/Desktop/EE 443/gth/test_audio/lily_audio';
audio_array = cell(1,numAudio);
audio_new = cell(1,numAudio);
audio_new_norm = cell(1,numAudio);
audio_freq = zeros(1,numAudio);
freq_new = zeros(1,numAudio);
audio_files=dir(fullfile(folder,'*.wav'));

MFCCs = cell(1,numAudio);
FBEs = cell(1,numAudio);
frames = cell(1,numAudio);
fileID = fopen('data.txt','w');
fileID2 = fopen('label.txt','w');

for k=1:numel(audio_files)
% for k=101:200
    % Resamble data to same length
    filename = audio_files(k).name;
    [audio_array{k}, audio_freq(k)] = audioread(filename);
%     if audio_freq(k) ~= 8000
%         audio_freq(k) = 8000;
%     end
    freq_new = length(audio_array{k})/0.4182;
    [p,q] = rat(freq_new/audio_freq(k),0.0001);
    audio_new{k} = resample(audio_array{k}(:,1),q,p);
    
    % Label data
    number = str2double(filename(1));
    fprintf(fileID2,"%d\n",number);
    
    % Normalize input audio
    max_unit = max(abs(audio_new{k}));
    audio_new_norm{k} = audio_new{k}/max_unit;
    
    % Calculate MFCCs
    N = round(audio_freq(k)*A); P = round(audio_freq(k)*(A-B));
    Y =  vec2frames(audio_new_norm{k},N,P);
    [ MFCCs{k}, FBEs{k}, frames{k}] = mfcc(audio_new_norm{k}(:,1), audio_freq(k), 25, 10, alpha, ones(25,1), R, M, C, L);
    
    newCoeff = [];
    newMFCC = zeros(39,1);
    for j = 1:40
        coeff = MFCCs{1,k}(:,j);
        delta = gradient(coeff);
        deltaDelta = gradient(delta);
        newCoeff = [newCoeff [coeff; delta; deltaDelta]];
    end
    for i = 1:39
       newMFCC(i,1) = max(newCoeff(i,:)); 
    end
    MFCCs{1,k} = newMFCC;
    
    format long
    fprintf(fileID,'%.8f\n',MFCCs{k}(:));
    fprintf(fileID,'\n');
end

%% 39*40

% for i = 1:numAudio
%     newCoeff = [];
%     newMFCC = zeros(39,1);
%     for j = 1:40
%         coeff = MFCCs{1,i}(:,j);
%         delta = gradient(coeff);
%         deltaDelta = gradient(delta);
%         newCoeff = [newCoeff [coeff; delta; deltaDelta]];
%     end
%     for k = 1:39
%        newMFCC(k,1) = max(newCoeff(k,:)); 
%     end
%     MFCCs{1,i} = newMFCC;
% end

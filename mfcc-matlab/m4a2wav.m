% folder = '/Users/FuYiran/Desktop/EE 443/gth/zx_audio';
% audio_array = cell(1,10);
% audio_files=dir(fullfile(folder,'*.m4a'));


m4AFilename = '/Users/FuYiran/Desktop/EE 443/gth/9.m4a';
[y,Fs] = audioread(m4AFilename);
wavFilename = '9.wav';
audiowrite(wavFilename,y,Fs);

% for i = 1:10
%    [y, Fs] = audio_array(i);
%    wavFilename = int2str(i) + 'new.wav';
%    audiowrite(wavFilename,y,Fs);
% end
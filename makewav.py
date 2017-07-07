import librosa
import fnmatch
import os
import re
import numpy as np
import scipy
from numpy import random

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'

sample_rate=16000
sav_n_secs=5

train_data_num=12000

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    #print (files)
    return files

def mix(audio_1, audio_2, sr, sav_n_secs,path_out,index):
  snr = random.randint(5,10)
  weight=np.power(10,(snr/20))

  file_mix="%07d_%02d.wav"%(index,snr)

  audio_1 = weight* audio_1
  audio_2 = audio_2

  audio_mix = (audio_1+audio_2)/2

  outfile = os.path.join(path_out,file_mix)
  librosa.output.write_wav(outfile, np.concatenate((audio_1,audio_2,audio_mix),axis=0), sr)


def main():
    outdir = 'mix'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    audio_total1, sr = librosa.load('./cao.wav', sr=sample_rate, mono=True)
    audio_total2, sr = librosa.load('./huang.wav', sr=sample_rate, mono=True)

    seglen = int(sav_n_secs * sr)

    len1 = audio_total1.shape[0] - seglen
    len2 = audio_total2.shape[0] - seglen

    for i in range(train_data_num):
      if i % 100 == 0:
        print(i)
      idx1=random.randint(0, len1)
      idx2=random.randint(0, len2)
      mix(audio_total1[idx1:idx1+seglen], audio_total2[idx2:idx2+seglen], sample_rate, sav_n_secs,outdir,i)

if __name__ == '__main__':
    main()

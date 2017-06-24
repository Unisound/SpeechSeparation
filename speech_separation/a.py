
from util import (cosSimilar,stft,istft,irstft)


import numpy as np
import scipy

fs= 16000
framesz= 0.032
hop= framesz*0.5


filename = "./0000000_09.wav"
_, audio = scipy.io.wavfile.read(filename)
print(audio.shape)
audio = audio.reshape(-1, 1)
print(audio.shape)


X, X_hlf=stft(audio, fs, framesz, hop)
amplitude= scipy.absolute(X_hlf)
angle= np.angle(X_hlf)
print(amplitude.shape)
print(angle.shape)


output_re = amplitude*np.cos(angle) + 1j*amplitude*np.sin(angle)
output_re=np.column_stack((output_re,np.conj(output_re[:,1:-1].T[::-1].T)))
outsize = angle.shape[0]*256
print("outsize:", outsize)
x_r=istft(output_re, fs, outsize, hop)
scipy.io.wavfile.write("spk.wav", fs, x_r)

import librosa
import numpy as np
from numpy import linalg 
import scipy

def cosSimilar(a,b):
  A=np.mat(a)  
  B=np.mat(b) 
  num = float(A * B.T)
  denom = linalg.norm(A) * linalg.norm(B)  
  cos = num / denom
  sim = 0.5 + 0.5 * cos
  return sim

def stft(x, fs, framesz, hop):
    """
     x - signal
     fs - sample rate
     framesz - frame size
     hop - hop size (frame size = overlap + hop size)
    """
    audio=np.reshape(x, (-1))
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fftpack.fft(w*audio[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X, X[:,:X.shape[1]/2+1]

def istft(X, fs, T, hop):
    #x = scipy.zeros(T*fs)
    x = scipy.zeros(T)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.fftpack.ifft(X[n]))
    return x

def irstft(X, fs, T, hop):
    #x = scipy.zeros(T*fs)
    x = scipy.zeros(T)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.fftpack.irfft(X[n]))
    return x

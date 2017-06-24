import numpy as np
import librosa

fs= 16000
framesz= 0.032
hop= framesz*0.5


filename = "./0000000_09.wav"
audio, sr = librosa.load(filename, sr=fs, mono=True)
print(audio.shape)


D =librosa.core.stft(audio,512, 256)
amplitude= np.abs(D)
angle= np.angle(D)
print(amplitude.shape)
print(angle.shape)

output = amplitude * np.exp(1j * angle)

x_r=librosa.core.istft(output, 256, 512)
librosa.output.write_wav("spk.wav", x_r,fs)

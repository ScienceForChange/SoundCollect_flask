
import maad
from maad import spl, sound
import waveform_analysis

response = {}

Gain = 40
sensitivity = -29.12
# time by seconds
DT = 1
VADC = 2
dBref = 94

# load audio file
w, fs = maad.sound.load('./audio/Oficina-X.WAV')

# calculate LeqT
LeqT = maad.spl.wav2leq(w, fs, gain=Gain, Vadc=VADC, dt=DT, sensitivity=sensitivity, dBref = dBref)

# calculate Lmin
Lmin = min(LeqT)

# calculate Lmax
Lmax = max(LeqT)

# calculare Leq
Leq = maad.util.mean_dB(LeqT)

# sort the median array to get L90 and L10
sorted_median_array = sorted(LeqT)
L90 = sorted_median_array[int(len(LeqT) * 0.1)]
L10 = sorted_median_array[int(len(LeqT) * 0.9)]

# calculate sharpness
# s, fs = maad.sound.load('./audio/Oficina-X.WAV')
# Sxx_power,_,_,_ = maad.sound.spectrogram (w, fs)
# sharp = maad.sound.sharpness(Sxx_power)

# calculate roughness
# rough = maad.sound.roughness(p)

# create a response
# add Lmin
response['Lmin'] = Lmin
# add Lmax
response['Lmax'] = Lmax
# add Leq
response['Leq'] = Leq
# add LAeq,T
response['LeqT'] = LeqT
# add L90
response['L90'] = L90
# add L10
response['L10'] = L10
# add sharpness
# response['sharpness'] = sharp

print(response)


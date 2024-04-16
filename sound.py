import maad
from maad import spl, sound
import numpy as np

response = []

# input sound file ================================================================================
w, fs = maad.sound.load('Vocales-Z.wav') 
p = maad.spl.wav2pressure(wave=w, gain=0)
maad.spl.pressure2dBSPL(abs(p))

# calculare Leq ===================================================================================
p_rms = maad.util.rms(p)
maad.spl.pressure2dBSPL(p_rms) 

x =  maad.spl.pressure2dBSPL(abs(p))

# print(len(x))

# split x into 1000 arrays, on a sample 1 minute WAV audio it had 2.8 milion values ===============
splitted_x_into_1000_arrays = np.array_split(x, 8)

# create a 1000 values median array ===============================================================
median_array = []
for i in splitted_x_into_1000_arrays:
    median_array.append(float(np.mean(i)))

# sort the median array to get L90 and L10 ========================================================
sorted_median_array = sorted(median_array)
L90 = sorted_median_array[int(len(median_array)*0.1)]
L10 = sorted_median_array[int(len(median_array)*0.9)]

# create a response ===============================================================================
# add Lmin
response.append('Lmin: ' + str(min(median_array)) + ' dB')
# add Lmax
response.append('Lmax: ' + str(max(median_array)) + ' dB')
# add Leq
response.append('Leq: ' + str(maad.spl.pressure2dBSPL(p_rms)) + ' dB')
# add L90
response.append('L90: ' + str(L90) + ' dB')
# add L10
response.append('L10: ' + str(L10) + ' dB')
# add LAeq,T
response.append('LAeq,T: ' + str(median_array))

print(response)





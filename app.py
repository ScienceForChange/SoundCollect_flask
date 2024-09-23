import os
import glob
import maad
import sys
import json
import numpy as np
from maad import spl, sound
from flask import Flask, render_template, request

import maad.sound
import maad.spl
import maad.features
import scipy
from numpy import pi, polymul
from scipy.signal import bilinear
import waveform_analysis
import matplotlib.pyplot as plt

from numpy import pi, log10
from scipy.signal import zpk2tf, zpk2sos, freqs, sosfilt
# from waveform_analysis.weighting_filters._filter_design import _zpkbilinear

from flask import Flask
from flask_cors import CORS

import random

# sys.path.append('..')

from mosqito.utils import load
from mosqito.sq_metrics import roughness_dw, roughness_dw_freq

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from mosqito.sq_metrics import loudness_zwst
from mosqito.sq_metrics import loudness_zwtv
from mosqito.sq_metrics import loudness_zwst_perseg
from mosqito.sq_metrics import sharpness_din_st
from mosqito.sq_metrics import sharpness_din_perseg
from mosqito.sq_metrics import sharpness_din_from_loudness
from mosqito.sq_metrics import sharpness_din_freq



# from A_weighting import A_weighting
from numpy import sum, log10, abs, mean, sqrt
# import librosa
# _MIN_ = sys.float_info.min
# sys.path.append('..')
from mosqito.sound_level_meter import noct_spectrum


app = Flask(__name__)
CORS(app)


@app.route("/test")
def main():
    return "You're home! ."


@app.route('/calibrate', methods=['POST'])
def calibrate():

    response = {}

    if request.method == 'POST':

        f = request.files['uploaded_file']

        f.save(f"./audio_calibrated/{request.files['uploaded_file'].filename}")

    calibrated_audio = {}

    gain = 40
    Vadc = 2
    dt = 1
    sensitivity = -38
    dBref = 94

    # load audio file
    w, fs = maad.sound.load(f"./audio_calibrated/{request.files['uploaded_file'].filename}")

    #apply "A" weighting filter to .wav signal
    signal_with_a_weighting = waveform_analysis.A_weight(w, fs)
 
    #Obtein the equivalent values over time from the "A" weighted signal
    LAeqT = maad.spl.wav2leq(wave=signal_with_a_weighting, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)

    # calculare Leq
    LAeq = maad.util.mean_dB(LAeqT)

    # return response
    response['calibrated_value'] = LAeq

    return response


@app.route('/audio')
def audio():

    response = {}

    # input sound file
    w, fs = maad.sound.load('./audio/audio_to_process.WAV')
    p = maad.spl.wav2pressure(wave=w, gain=0)
    maad.spl.pressure2dBSPL(abs(p))

    # calculare Leq
    p_rms = maad.util.rms(p)
    maad.spl.pressure2dBSPL(p_rms)

    x =  maad.spl.pressure2dBSPL(abs(p))

    #split x into 1000 arrays, on a sample 1 minute W AV audio it had 2.8 milion values
    splitted_x_into_1000_arrays = np.array_split(x, 10)

    # create a 1000 values median array
    median_array = []
    for i in splitted_x_into_1000_arrays:
        median_array.append(float(np.mean(i)))

    # sort the median array to get L90 and L10
    sorted_median_array = sorted(median_array)
    L90 = sorted_median_array[int(len(median_array)*0.1)]
    L10 = sorted_median_array[int(len(median_array)*0.9)]

    # get loudness
    # s, fs = maad.sound.load('./audio/Oficina-X.WAV')
    Sxx_power,_,_,_ = maad.sound.spectrogram (w, fs)
    sharp = maad.sound.sharpness(Sxx_power)

    # calculate roughness
    # rough = maad.sound.roughness(w)

    # create a response
    # add Lmin
    response['Lmin'] = min(median_array)
    # add Lmax
    response['Lmax'] = max(median_array)
    # add Leq
    response['Leq'] = maad.spl.pressure2dBSPL(p_rms)
    # add LAeq,T
    response['LAeqT'] = median_array
    # add L90
    response['L90'] = L90
    # add L10
    response['L10'] = L10
    # add sharpness
    response['sharpness'] = sharp

    # return response
    return json.dumps(response)


@app.route('/audio_new/<coeficiente_calibracion>')
def audio_new(coeficiente_calibracion):

    coeficiente_calibracion = float(coeficiente_calibracion)
    gain = 40
    Vadc = 2
    dt = 1
    sensitivity = -38
    dBref = 94

    #Load the .wav file
    signal, fs = maad.sound.load('./audio/audio_to_process.WAV')

    #apply "A" weighting filter to .wav signal
    signal_with_a_weighting = waveform_analysis.A_weight(signal, fs)
 
    #Obtein the equivalent values over time from the "A" weighted signal
    LAeqT = maad.spl.wav2leq(wave=signal_with_a_weighting, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)

    # calculare Leq
    LAeq = maad.util.mean_dB(LAeqT)

    # apply correction from user input value
    LAeqT = LAeqT - coeficiente_calibracion
    LAeq = LAeq - coeficiente_calibracion

    # calculate Lmin
    LAmin = min(LAeqT)

    # calculate Lmax
    LAmax = max(LAeqT)

    # sort LAeqT array (for L10 and L90)
    sorted_median_array = sorted(LAeqT)

    # calculate L90 and L10
    L90 = sorted_median_array[int(len(LAeqT) * 0.1)]
    L10 = sorted_median_array[int(len(LAeqT) * 0.9)]

    # create a response
    response = {}

    response['LAeqT'] = LAeqT.tolist()
    response['Leq'] = LAeq
    response['Lmin'] = LAmin
    response['Lmax'] = LAmax
    response['L90'] = L90
    response['L10'] = L10

    audio_file ='./audio/audio_to_process.WAV'

    # calculate fluctuation strength  from audio file =============================================
    sig, fs = load(audio_file, wav_calib=2 * 2 ** 0.5)
    #  compute acoustic Loudness according to Zwicker method for stationary signals
    N, N_specific, bark_axis = loudness_zwst(sig, fs, field_type="free")
    F = acousticFluctuation(N_specific, fmod=4)
    response['fluctuation'] = F

    # calculate sharpness =========================================================================
    sharpness = sharpness_din_st(sig, fs, weighting="din")
    response['sharpness'] = sharpness

    # calculate loudness ==========================================================================
    # compute acoustic Loudness according to Zwicker method for stationary signals
    N, N_specific, bark_axis = loudness_zwst(sig, fs, field_type="free")
    response['loudness'] = N

    #  # calculate audio roughness ================================================================
    #  # takes too much time/CPU to get roughnes parameter 
    # sig, fs = maad.sound.load(audio_file )
    # r, r_spec, bark, time = roughness_dw(sig, fs, overlap=0)
    # response['roughness'] = np.mean(r)

    # calculate 1/3 octave ========================================================================
    # Frequency analysis: Use noct_spectrum with signal weighted A
    spec_3_A, freq_3 = noct_spectrum(signal_with_a_weighting, fs, fmin=50, fmax=16000, n=3)
    spec_3_sin_A, freq_3 = noct_spectrum(signal, fs, fmin=50, fmax=16000, n=3)

    spec_3_dBA = 20 * np.log10(spec_3_A / 2e-5)
    spec_3_dB = 20 * np.log10(spec_3_sin_A / 2e-5)

    # clean spec_3 array
    spec_3_A = [item[0] for item in spec_3_A]
    
    # clean spec_3_dB array
    spec_3_dBA = [item[0] for item in spec_3_dBA]

    # clean spec_3 array
    spec_3_sin_A = [item[0] for item in spec_3_sin_A]

    # clean spec_3_dB array
    spec_3_dB = [item[0] for item in spec_3_dB]

    # add 1/3 octave to response for x-axis
    response['freq_3'] = freq_3.tolist()
    # add 1/3 octave to response for y-axis without ponderation
    response['spec_3'] = spec_3_dBA
    # add 1/3 octave to response for y-axis with ponderation
    response['spec_3_dB'] = spec_3_dB


    return response


@app.route('/convert_audio_into_parameters/<coeficiente_calibracion>', methods=['POST'])
def convert_audio_into_parameters(coeficiente_calibracion):

    # if request.method == 'POST':

    file = request.files['uploaded_file']

    # Create random number to uniquely identify each audio with different filename to evaluating them later.
    # This way each sound will be separated during the process from each other, right now it rewrites the new sound every time user uploads a audio for processing
    random_number = random.randint(1, 100000)

    # file needs to be saved before applying any calculating process
    file.save(f"./audio_calibrated/{random_number}_{request.files['uploaded_file'].filename}")

    coeficiente_calibracion = float(coeficiente_calibracion)
    gain = 40
    Vadc = 2
    dt = 1
    sensitivity = -38
    dBref = 94

    #Load the .wav file
    signal, fs = maad.sound.load(f"./audio_calibrated/{random_number}_{request.files['uploaded_file'].filename}")

    #apply "A" weighting filter to .wav signal
    signal_with_a_weighting = waveform_analysis.A_weight(signal, fs)

    #Obtein the equivalent values over time from the "A" weighted signal
    LAeqT = maad.spl.wav2leq(wave=signal_with_a_weighting, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)

    # calculate Leq
    LAeq = maad.util.mean_dB(LAeqT)

    # apply correction from user input value
    LAeqT = LAeqT - coeficiente_calibracion
    LAeq = LAeq - coeficiente_calibracion

    # calculate Lmin
    LAmin = min(LAeqT)

    # calculate Lmax
    LAmax = max(LAeqT)

    # sort LAeqT array (for L10 and L90)
    sorted_median_array = sorted(LAeqT)

    # calculate L90 and L10
    L90 = sorted_median_array[int(len(LAeqT) * 0.1)]
    L10 = sorted_median_array[int(len(LAeqT) * 0.9)]

    # initialize response and response_data
    response = {}
    response_data = {}

    response_data['LAeqT'] = [round(value, 1) for value in LAeqT.tolist()]
    response_data['Leq'] = round(LAeq, 2)
    response_data['Lmin'] = round(LAmin, 2)
    response_data['Lmax'] = round(LAmax, 2)
    response_data['L90'] = round(L90, 2)
    response_data['L10'] = round(L10, 2)

    audio_file = f"./audio_calibrated/{random_number}_{request.files['uploaded_file'].filename}"

    # calculate fluctuation strength  from audio file =============================================
    sig, fs = load(audio_file, wav_calib=2 * 2 ** 0.5)
    #  compute acoustic Loudness according to Zwicker method for stationary signals
    N, N_specific, bark_axis = loudness_zwst(sig, fs, field_type="free")
    F = acousticFluctuation(N_specific, fmod=4)
    response_data['fluctuation'] = F

    # calculate sharpness =========================================================================
    sharpness = sharpness_din_st(sig, fs, weighting="din")
    response_data['sharpness'] = sharpness

    # calculate loudness ==========================================================================
    # compute acoustic Loudness according to Zwicker method for stationary signals
    N, N_specific, bark_axis = loudness_zwst(sig, fs, field_type="free")
    response_data['loudness'] = N

    #  # calculate audio roughness ================================================================
    #  # takes too much time/CPU to get roughnes parameter 
    # sig, fs = maad.sound.load(audio_file )
    # r, r_spec, bark, time = roughness_dw(sig, fs, overlap=0)
    # response['roughness'] = np.mean(r)

    # calculate 1/3 octave ========================================================================
    # Frequency analysis: Use noct_spectrum with signal weighted A
    spec_3_A, freq_3 = noct_spectrum(signal_with_a_weighting, fs, fmin=50, fmax=16000, n=3)
    spec_3_sin_A, freq_3 = noct_spectrum(signal, fs, fmin=50, fmax=16000, n=3)

    spec_3_dBA = 20 * np.log10(spec_3_A / 2e-5)
    spec_3_dB = 20 * np.log10(spec_3_sin_A / 2e-5)

    # clean spec_3 array
    spec_3_A = [item[0] for item in spec_3_A]
    
    # clean spec_3_dB array
    spec_3_dBA = [item[0] for item in spec_3_dBA]

    # clean spec_3 array
    spec_3_sin_A = [item[0] for item in spec_3_sin_A]

    # clean spec_3_dB array
    spec_3_dB = [item[0] for item in spec_3_dB]

    # add 1/3 octave to response for x-axis
    response_data['freq_3'] = freq_3.tolist()
    # add 1/3 octave to response for y-axis without ponderation
    response_data['spec_3'] = spec_3_dBA
    # add 1/3 octave to response for y-axis with ponderation
    response_data['spec_3_dB'] = spec_3_dB

    # calculate spec_3_dBC value ==================================================================
    # LZeq = maad.spl.wav2leq(wave=signal, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)

    C_weighting_values = np.array([-1.3, -0.8, -0.5, -0.3, -0.2, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3, -4.4, -6.2, -8.6])

    #Frequency analysis: Use noct_spectrum with signal weighted A
    f_start = 50
    f_final = 16000
    specZ_3, freq_3 = noct_spectrum(signal, fs, f_start, f_final, n=3)

    spec_3_dBZ = 20 * np.log10(specZ_3 / 2e-5)
    spec_3_dBC = spec_3_dBZ + C_weighting_values
    
    # original version of the code, but it shows list within the list of spec_3_dBC variable
    # response_data['spec_3_dBC'] = spec_3_dBC
     
    # calculate median values for spec_3_dBC
    spec_3_dBC_median = [np.median(sublist) for sublist in spec_3_dBC]
    response_data['spec_3_dBC'] = spec_3_dBC_median


    # create response =============================================================================
    response['status'] = 'success'
    response['data'] = response_data

    # remove audio file after processing it
    os.remove(f"./audio_calibrated/{random_number}_{request.files['uploaded_file'].filename}")

    return response


def ABC_weighting(curve='A'):
    """
    Design of an analog weighting filter with A, B, or C curve.

    Returns zeros, poles, gain of the filter.

    Examples
    --------
    Plot all 3 curves:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> for curve in ['A', 'B', 'C']:
    ...     z, p, k = ABC_weighting(curve)
    ...     w = 2*pi*logspace(log10(10), log10(100000), 1000)
    ...     w, h = signal.freqs_zpk(z, p, k, w)
    ...     plt.semilogx(w/(2*pi), 20*np.log10(h), label=curve)
    >>> plt.title('Frequency response')
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.ylim(-50, 20)
    >>> plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    >>> plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
    >>> plt.legend()
    >>> plt.show()

    """
    if curve not in 'ABC':
        raise ValueError('Curve type not understood')

    # ANSI S1.4-1983 C weighting
    #    2 poles on the real axis at "20.6 Hz" HPF
    #    2 poles on the real axis at "12.2 kHz" LPF
    #    -3 dB down points at "10^1.5 (or 31.62) Hz"
    #                         "10^3.9 (or 7943) Hz"
    #
    # IEC 61672 specifies "10^1.5 Hz" and "10^3.9 Hz" points and formulas for
    # derivation.  See _derive_coefficients()

    z = [0, 0]
    p = [-2*pi*20.598997057568145,
         -2*pi*20.598997057568145,
         -2*pi*12194.21714799801,
         -2*pi*12194.21714799801]
    k = 1

    if curve == 'A':
        # ANSI S1.4-1983 A weighting =
        #    Same as C weighting +
        #    2 poles on real axis at "107.7 and 737.9 Hz"
        #
        # IEC 61672 specifies cutoff of "10^2.45 Hz" and formulas for
        # derivation.  See _derive_coefficients()

        p.append(-2*pi*107.65264864304628)
        p.append(-2*pi*737.8622307362899)
        z.append(0)
        z.append(0)

    elif curve == 'B':
        # ANSI S1.4-1983 B weighting
        #    Same as C weighting +
        #    1 pole on real axis at "10^2.2 (or 158.5) Hz"

        p.append(-2*pi*10**2.2)  # exact
        z.append(0)

    # TODO: Calculate actual constants for this
    # Normalize to 0 dB at 1 kHz for all curves
    b, a = zpk2tf(z, p, k)
    k /= abs(freqs(b, a, [2*pi*1000])[1][0])

    return np.array(z), np.array(p), k


def A_weighting(fs, output='ba'):
    """
    Design of a digital A-weighting filter.

    Designs a digital A-weighting filter for
    sampling frequency `fs`.
    Warning: fs should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.

    Parameters
    ----------
    fs : float
        Sampling frequency
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.

    Examples
    --------
    Plot frequency response

    >>> from scipy.signal import freqz
    >>> import matplotlib.pyplot as plt
    >>> fs = 200000
    >>> b, a = A_weighting(fs)
    >>> f = np.logspace(np.log10(10), np.log10(fs/2), 1000)
    >>> w = 2*pi * f / fs
    >>> w, h = freqz(b, a, w)
    >>> plt.semilogx(w*fs/(2*pi), 20*np.log10(abs(h)))
    >>> plt.grid(True, color='0.7', linestyle='-', which='both', axis='both')
    >>> plt.axis([10, 100e3, -50, 20])

    Since this uses the bilinear transform, frequency response around fs/2 will
    be inaccurate at lower sampling rates.
    """
    z, p, k = ABC_weighting('A')

    # Use the bilinear transformation to get the digital filter.
    z_d, p_d, k_d = _zpkbilinear(z, p, k, fs)

    if output == 'zpk':
        return z_d, p_d, k_d
    elif output in {'ba', 'tf'}:
        return zpk2tf(z_d, p_d, k_d)
    elif output == 'sos':
        return zpk2sos(z_d, p_d, k_d)
    else:
        raise ValueError("'%s' is not a valid output form." % output)


def A_weight(signal, fs):
    """
    Return the given signal after passing through a digital A-weighting filter

    signal : array_like
        Input signal, with time as dimension
    fs : float
        Sampling frequency
    """
    # TODO: Upsample signal high enough that filter response meets Type 0
    # limits.  A passes if fs >= 260 kHz, but not at typical audio sample
    # rates. So upsample 48 kHz by 6 times to get an accurate measurement?
    # TODO: Also this could just be a measurement function that doesn't
    # save the whole filtered waveform.
    sos = A_weighting(fs, output='sos')
    return sosfilt(sos, signal)


def acousticFluctuation(specificLoudness, fmod=4):
    specificLoudnessdiff = np.zeros(len(specificLoudness))
    for i in range(len(specificLoudness)):
        if i == 0:
            specificLoudnessdiff[i] = specificLoudness[i]
        else:
            specificLoudnessdiff[i] = abs(specificLoudness[i] - specificLoudness[i - 1])
    F = (0.008 * sum(0.1 * specificLoudnessdiff)) / ((fmod / 4) + (fmod / 4))
    return F


def _derive_coefficients():
    """
    Calculate A- and C-weighting coefficients with equations from IEC 61672-1

    This is for reference only. The coefficients were generated with this and
    then placed in ABC_weighting().
    """
    import sympy as sp

    # Section 5.4.6
    f_r = 1000
    f_L = sp.Pow(10, sp.Rational('1.5'))  # 10^1.5 Hz
    f_H = sp.Pow(10, sp.Rational('3.9'))  # 10^3.9 Hz
    D = sp.sympify('1/sqrt(2)')  # D^2 = 1/2

    f_A = sp.Pow(10, sp.Rational('2.45'))  # 10^2.45 Hz

    # Section 5.4.9
    c = f_L**2 * f_H**2
    b = (1/(1-D))*(f_r**2+(f_L**2*f_H**2)/f_r**2-D*(f_L**2+f_H**2))

    f_1 = sp.sqrt((-b - sp.sqrt(b**2 - 4*c))/2)
    f_4 = sp.sqrt((-b + sp.sqrt(b**2 - 4*c))/2)

    # Section 5.4.10
    f_2 = (3 - sp.sqrt(5))/2 * f_A
    f_3 = (3 + sp.sqrt(5))/2 * f_A

    # Section 5.4.11
    assert abs(float(f_1) - 20.60) < 0.005
    assert abs(float(f_2) - 107.7) < 0.05
    assert abs(float(f_3) - 737.9) < 0.05
    assert abs(float(f_4) - 12194) < 0.5

    for f in ('f_1', 'f_2', 'f_3', 'f_4'):
        print('{} = {}'.format(f, float(eval(f))))

    # Section 5.4.8  Normalizations
    f = 1000
    C1000 = (f_4**2 * f**2)/((f**2 + f_1**2) * (f**2 + f_4**2))
    A1000 = (f_4**2 * f**4)/((f**2 + f_1**2) * sp.sqrt(f**2 + f_2**2) *
                             sp.sqrt(f**2 + f_3**2) * (f**2 + f_4**2))

    # Section 5.4.11
    assert abs(20*log10(float(C1000)) + 0.062) < 0.0005
    assert abs(20*log10(float(A1000)) + 2.000) < 0.0005

    for norm in ('C1000', 'A1000'):
        print('{} = {}'.format(norm, float(eval(norm))))


def script(gain=40):
    #Constants
    # gain= (28.8+12)
    Vadc = 2
    dt = 1
    sensitivity = -38
    #sensitivity = -45
    dBref = 94

    #Load the .wav file
    w, fs = maad.sound.load('Audio Bcn/ext1-max.berned.WAV')
    #p = maad.spl.wav2leq(wave=w, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)
    #p = maad.spl.wav2dBSPL(w, gain, Vadc, sensitivity, dBref, pRef=2e-05)

    #apply A weighting filter to .wav signal
    # w = waveform_analysis.A_weight(w, fs)
    rms_value = np.sqrt(np.mean(np.abs(w) ** 2))
    print(len(w))
    #print(weighted_signal)
    #Obtein the equivalent values from the A weighted signal
    p = maad.spl.wav2leq(wave=w, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)
    print(len(p))
    print(p)
    print('LAeq from wav', maad.util.mean_dB(p))
    plt.figure(len(p))
    plt.plot(p)

    return plt.show()


def temp_new_a_weight():
        #Constants
    gain= 2.8 #if sensitivity=-38 then gain=40.8
    Vadc = 2
    dt = 1
    sensitivity = 0 # if sensitivity=0 then gain=2.8
    dBref = 94
    coef_calib=0 #Valor SPL_APP - SPL_Sonometro. Usar este valor para ajustar LAeq (y posteriores).
    flag_calib1=0
    flag_calib2=0
    want_to_calibrate=input("¿Quieres calibrar?")
    match want_to_calibrate:
        case "yes":
            is_sonometer = input("¿Tienes un sonometro?")
            match is_sonometer:
                case "yes":
                    user_dBA=input("Introduce the SPL (dBA) value: ")
                    user_dBA=float(user_dBA)
                    flag_calib1=1
                    flag_calib2=1
                case "no":
                    print("Measure in a silent room. It's supposed to measure 40dBA.")
                    flag_calib1 = 1
                    flag_calib2 = 0
        case "no":
            print("Ok, default values are used.")
            flag_calib1=0

    #Load the .wav file
    w, fs = maad.sound.load('Audios test/test-office-AA-iphone.wav')
    #w, fs = load('Audio Bcn/rosa-ortega.alsius.marta.WAV', wav_calib=2 * 2 **0.5)
    #w, fs = load('Audio Bcn/rosa-ortega.alsius.marta.WAV')

    #p = maad.spl.wav2leq(wave=w, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)
    #p = maad.spl.wav2dBSPL(w, gain, Vadc, sensitivity, dBref, pRef=2e-05)

    #apply A weighting filter to .wav signal
    wA = waveform_analysis.A_weight(w, fs)
    rms_value = np.sqrt(np.mean(np.abs(wA) ** 2))
    print(len(wA))
    #print(weighted_signal)

    #Obtein the equivalent values from the A weighted signal
    LAeq = maad.spl.wav2leq(wave=wA, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)
    LAeqT= maad.util.mean_dB(LAeq)
    if flag_calib1==1 and flag_calib2==1:
        coef_calib=LAeqT-user_dBA
        LAeq_calib=LAeq-coef_calib
        LAeqT_calib= LAeqT-coef_calib
    elif flag_calib1==1 and flag_calib2==0:
        coef_calib = LAeqT-40
        LAeq_calib = LAeq-coef_calib
        LAeqT_calib = LAeqT-coef_calib
    elif flag_calib1==0:
        coef_calib=0
        LAeq_calib = LAeq - coef_calib
        LAeqT_calib = LAeqT - coef_calib
    print("Wav file duration (in seconds): ",len(LAeq))
    #print(LAeq)
    print('LAeq from wav', LAeqT)
    print("LAeq calibrated: ",LAeqT_calib)
    print("Factor de corrección: ",coef_calib)

# app.config['MAX_CONTENT_LENGTH'] = 50 * 1000 * 1000

def acousticFluctuation(specificLoudness, fmod=4):
    specificLoudnessdiff = np.zeros(len(specificLoudness))
    for i in range(len(specificLoudness)):
        if i == 0:
            specificLoudnessdiff[i] = specificLoudness[i]
        else:
            specificLoudnessdiff[i] = abs(specificLoudness[i] - specificLoudness[i - 1])
    F = (0.008 * sum(0.1 * specificLoudnessdiff)) / ((fmod / 4) + (fmod / 4))
    return F

if __name__ == "__main__":
    # app.run(threaded=True)
    app.run(debug=True, threaded=True)

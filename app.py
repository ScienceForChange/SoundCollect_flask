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
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from numpy import pi, log10
from scipy.signal import zpk2tf, zpk2sos, freqs, sosfilt
from waveform_analysis.weighting_filters._filter_design import _zpkbilinear


app = Flask(__name__)

# @app.route("/")
# def main():
#     return "You're home!"

@app.route('/calibrate', methods=['POST'])
def calibrate():

    response = {}

    if request.method == 'POST':

        f = request.files['uploaded_file']

        f.save(f"./audio_calibrated/{request.files['uploaded_file'].filename}")

    calibrated_audio = {}

    Gain = 0
    sensitivity = -29.12
    # time by seconds
    DT = 1
    VADC = 2
    dBref = 94

    # load audio file
    w, fs = maad.sound.load(f"./audio_calibrated/{request.files['uploaded_file'].filename}")

    # calculate LeqT
    LeqT = maad.spl.wav2leq(w, fs, gain=Gain, Vadc=VADC, dt=DT, sensitivity=sensitivity, dBref = dBref)

    # calculare Leq
    Leq = maad.util.mean_dB(LeqT)

    # return response
    response['Leq'] = Leq

    return json.dumps(response)


@app.route('/audio')
def audio():

    response = {}

    # input sound file
    w, fs = maad.sound.load('./audio/Oficina-X.WAV')
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


@app.route('/audio_new')
def audio_new():

    response = {}

    Gain = 0
    sensitivity = -29.12
    dt = 1      # time in seconds
    vadc = 2
    dBref = 94

    # load audio file
    w, fs = maad.sound.load('./audio/Oficina-X.WAV')

    # calculate LeqT
    LeqT = maad.spl.wav2leq(w, fs, gain=Gain, Vadc=vadc, dt=dt, sensitivity=sensitivity, dBref = dBref)

    
    # Create a JSON Encoder class
    class json_serialize(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    # serialize LEQT to JSON
    leqt_json = json.dumps({'nums': LeqT}, cls=json_serialize)
    # leqt_json = json.dumps({'nums': response.tolist()})


    # calculate Lmin
    Lmin = min(LeqT)

    # calculate Lmax
    Lmax = max(LeqT)

    # calculare Leq
    Leq = maad.util.mean_dB(LeqT)

    # sort the median array to get L90 and L10
    sorted_median_array = sorted(LeqT)

    # calculate L90 and L10
    L90 = sorted_median_array[int(len(LeqT) * 0.1)]
    L10 = sorted_median_array[int(len(LeqT) * 0.9)]

    # create a response
    response['Lmin'] = Lmin
    response['Lmax'] = Lmax
    response['Leq'] = Leq
    # response['LeqT'] = leqt_json
    response['L90'] = L90
    response['L10'] = L10

    # return np.array(LeqT)

    # return json.dumps(response)
    return LeqT


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
    w = waveform_analysis.A_weight(w, fs)
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


    #print('Leq from volt', maad.util.mean_dB(weighted_signal))
    #result = 20 * np.log10(rms_value)
    #print(result)

@app.route('/temp')
def temp(gain=10):
    
    # gain=40
    Vadc = 2
    dt = 1
    sensitivity = -38
    dBref = 94

    #Load the .wav file
    signal, fs = maad.sound.load('./audio/Oficina-X.WAV')

    #apply "A" weighting filter to .wav signal
    signal_with_a_weighting = waveform_analysis.A_weight(signal, fs)
 
    #Obtein the equivalent values over time from the "A" weighted signal
    LAeqT = maad.spl.wav2leq(wave=signal, f=fs, gain=gain, Vadc=Vadc, dt=dt, sensitivity=sensitivity, dBref=dBref)

    # calculate Lmin
    LAmin = min(LAeqT)

    # calculate Lmax
    LAmax = max(LAeqT)

    # calculare Leq
    LAeq = maad.util.mean_dB(LAeqT)

    # sort LAeqT array (for L10 and L90)
    sorted_median_array = sorted(LAeqT)

    # calculate L90 and L10
    L90 = sorted_median_array[int(len(LAeqT) * 0.1)]
    L10 = sorted_median_array[int(len(LAeqT) * 0.9)]

    # create a response
    response = {}

    response['LAeqT'] = LAeqT.tolist()
    response['LAeq'] = LAeq
    response['LAmin'] = LAmin
    response['LAmax'] = LAmax
    response['L90'] = L90
    response['L10'] = L10

    return response


if __name__ == "__main__":
    app.run(debug=True)


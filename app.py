import os
import glob
import maad
import sys
import json
import numpy as np
from maad import spl, sound
from flask import Flask, render_template, request

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

    return json.dumps(response)




if __name__ == "__main__":
    app.run(debug=True)


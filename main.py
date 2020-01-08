from os import listdir
from os.path import isfile, join
from sys import argv
import numpy as np
from numpy.fft import fft
from scipy import signal as ss
from scipy.io import wavfile
from scipy.signal import butter
from sklearn.linear_model import LogisticRegression


def train():
    data = load_data()
    labels, parameters = raw_data_to_params(data)
    model = LogisticRegression().fit(parameters, labels)
    display_training_loss(labels, model, parameters)
    return model


def display_training_loss(labels, model, parameters):
    predict = model.predict(parameters)
    print("Learning loss:", end=' ')
    print(1 - (sum([predict[i] == labels[i] for i in range(len(labels))]) / len(labels)))
    MasK = sum([int(predict[i] == 0 and labels[i] == 1) for i in range(len(labels))])
    KasM = sum([int(predict[i] == 1 and labels[i] == 0) for i in range(len(labels))])
    KK = sum([int(predict[i] == 0 == labels[i]) for i in range(len(labels))])
    MM = sum([int(predict[i] == 1 == labels[i]) for i in range(len(labels))])
    print(f"Correctly recognized {MM} men, {KK} woman. {MasK} men recognized as woman, and {KasM} woman recognized as men.")


def raw_data_to_params(data):
    labels = []
    parameters = []
    for i in range(len(data)):
        label, fs, audio = data[i]
        labels.append(int(label == 'M'))
        audio, f = process_audio(fs, audio)
        parameters.append([f, np.quantile(audio, 0.8), np.median(audio)])
    parameters = np.array(parameters)
    return labels, parameters


def load_data():
    files = [join("train", f) for f in listdir("train") if isfile(join("train", f))]
    data = [[filename[-5], *wavfile.read(filename)] for filename in files]
    return data


def predict(model, filepath):
    fs, audio = wavfile.read(filepath)
    audio, f = process_audio(fs, audio)
    parameters = [[f, np.median(audio)]]
    return "M" if model.predict(parameters)[0] else "K"


def process_audio(fs, audio):
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    audio = bandpass_filter(audio, fs)
    audio = fourier(audio)
    f = calculate_root_freq(audio, fs)
    return audio, f


def calculate_root_freq(audio, fs):
    tmp = np.ndarray(shape=audio.shape, dtype=np.float64)
    np.copyto(tmp, audio)
    for k in range(2, 5):
        x = ss.decimate(audio, k)
        add_zeros = len(audio) - len(x)
        final = np.pad(x, (0, add_zeros), 'constant')
        tmp *= final
    # get max aplitude in range 65Hz to 300Hz
    arg_min = int(65 / fs * len(tmp))
    arg_max = int(300 / fs * len(tmp))
    tmp_sliced = tmp[arg_min:arg_max]
    f = (np.argmax(tmp_sliced) + arg_min) / len(tmp) * fs
    return f


def fourier(audio):
    n = len(audio)
    audio = audio * np.kaiser(n, 15)
    audio = fft(audio)
    audio = abs(audio) / n
    return audio


def bandpass_filter(audio, fs):
    nyquist_rate = fs / 2
    order = 5
    low = 80 / nyquist_rate
    high = 3000 / nyquist_rate
    sos = butter(order, [low, high], btype='band', output='sos')
    audio = ss.sosfilt(sos, audio)
    return audio


if __name__ == '__main__':
    model = train()
    print(predict(model, argv[-1]))
    # print(predict(model, "test/test_K.wav"))
    # print(predict(model, "test/test2_K.wav"))
    # print(predict(model, "test/test3_K.wav"))
    # print(predict(model, "test/test4_K.wav"))
    # print(predict(model, "test/test5_K.wav"))
    # print(predict(model, "test/test_M.wav"))
    # print(predict(model, "test/test2_M.wav"))

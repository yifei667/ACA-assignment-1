import numpy as np
from scipy.signal import find_peaks
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt  # For ploting
import os
import glob

def block_audio(x, blockSize, hopSize, fs):
    if isinstance(hopSize, int):
        move = blockSize - hopSize
        num = int(np.ceil((len(x) - blockSize) / move) + 1)
        total_num = (num - 1) * (blockSize - hopSize) + blockSize
        matrix = np.pad(x, (0, total_num - len(x)), 'constant')
        matrix1 = np.zeros((num, blockSize))
        for i in range(len(matrix1)):
            matrix1[i] = matrix[i * move:i * move + blockSize]
        end_index = move * num
        index = np.arange(0, end_index, move)
        time = index / fs
    else:
        num = len(hopSize)
        total_num = int(np.sum(hopSize) + blockSize[-1] - len(x))
        matrix = np.pad(x, (0, total_num), 'constant')
        maxBlock = int(max(blockSize))
        matrix1 = np.zeros((num, maxBlock))
        lastHop = 0
        time = np.zeros(len(hopSize))
        for index, item in enumerate(hopSize):
            item = int(item)
            block = int(blockSize[index])
            difference = maxBlock - block
            currentblock = x[lastHop:(lastHop + block)]
            currentblock = np.pad(currentblock, (0, difference), 'constant')
            matrix1[index,:] = currentblock
            time[index] = float(lastHop / fs)
            lastHop = lastHop + item - 1
    return matrix1, time


def comp_acf(inputVector, bIsNormalized):
    r = np.zeros(len(inputVector) - 1)
    for i in range(len(r)):
        for j in range(len(inputVector) - i):
            r[i] += inputVector[j] * inputVector[i + j]
    r = r * bIsNormalized / max(abs(r))
    return r


def get_f0_from_acf(r, fs):
    indice, dict1 = find_peaks(r)
    matrix = r[indice]
    list1 = matrix.argsort()[-2:][::-1]
    index1 = list1[0]
    index2 = list1[1]
    diff = abs(indice[index2]-indice[index1])
    # Avoiding dividing by zero
    if diff == 0:
        f0 = 0
    else:
        f0 = fs/diff
    return f0


def track_pitch_acf(x, blockSize, hopSize, fs):
    matrix, time = block_audio(x, blockSize, hopSize, fs)
    f0 = np.zeros(len(matrix))
    for i in range(len(matrix)):
        block = matrix[i]
        r = comp_acf(block, 1)
        f0[i] = get_f0_from_acf(r, fs)
        print(f0[i], i)
    return f0, time


# Creates 2 second sine wave, 1 second of 441 Hz, 1 second of 882 Hz
fs = 44100
f1 = 441
f2 = 882
x1 = np.arange(fs)
x2 = np.arange(fs, 2 * fs)
y1 = np.sin(2 * np.pi * f1 * (x1 / fs))
y2 = np.sin(2 * np.pi * f2 * (x2 / fs))
x = np.append(x1, x2)
y = np.append(y1, y2)
# Plots 200 samples, 100 samples of 441 (one cycle) and 100 samples of 882 (two cycles)
# plt.plot(x[44000:44200], y[44000:44200])
f0, time = track_pitch_acf(y, 1764, 882, fs)
# Finds RMS error in Hz
hz = np.append([441]*50, [882]*49)
err = np.sqrt((f0 - hz) ** 2)

# Plots generated f0, an array of the correct frequency, and the RMS error in hz.
plt.plot(time, f0, 'r--', time, err, 'g--')
plt.axis([0, 2, 0, 1000])
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('Testing ACF on the 441hz and 882hz sine')
plt.legend(["ACF estimated f0", "Actual frequency", "RMS Rrror (Hz)"])
plt.show()

# # Discussion
# The error for the sine wave was zero. This was lower than expected and lower than our original implementation.
# For get_f0_from_acf function, we improved upon scipy.signal.find_peaks. We only used primary and secondary peaks.
# #


def convert_freq2midi(freqInHz):
    if isinstance(freqInHz, float):
        freqInMIDI = int(np.around(69 + 12 * np.log2(freqInHz / 440)))
        return freqInMIDI
    elif isinstance(freqInHz, (np.ndarray, list)):
        freqInMIDI = np.zeros(np.shape(freqInHz))
        for index, item in enumerate(freqInHz):
            if not item.dtype == 'float64':
                if len(item) > 1:
                    for obj in freqInHz[index]:
                        if item == 0:
                            continue
                        else:
                            freqInMIDI[index] = int(np.around(69 + 12 * np.log2(obj / 440.0)))
                elif len(item) == 1:
                    if item == 0:
                        continue
                    else:
                        freqInMIDI[index] = int(np.around(69 + 12 * np.log2(item / 440.0)))
                else:
                    print("Unknown array size!")
            else:
                if item == 0:
                    continue
                else:
                    freqInMIDI[index] = int(np.around(69 + 12 * np.log2(item / 440.0)))
        return freqInMIDI
    elif isinstance(freqInHz, np.matrix):
        print('matrix')
        freqInMIDI = np.zeros(np.shape(freqInHz))
        for index, row in enumerate(freqInHz):
            for item in row:
                freqInMIDI[index, row] = int(np.around(69 + 12 * np.log2(item / 440.0)))
        return freqInMIDI
    else:
        print('Data type check failed.')


def eval_pitchtrack(estimateInHz, groundtruthInHz):
    # Compute RMS error in cents
    errCent = np.zeros(len(estimateInHz))
    for index, item in enumerate(estimateInHz):
        if groundtruthInHz[index] != 0:
            if item == 0:
                errCents = 1200 * np.log2(1 / groundtruthInHz[index])
            else:
                errCents = 1200 * np.log2(item / groundtruthInHz[index])
        else:
            continue
        errCent[index] = np.floor(np.abs(errCents))
    errCentRms = np.mean(errCent)
    return errCentRms


def run_evaluation(complete_path_to_data_folder):
    # Creates two dictionaries, one for wave files, one for text files.
    wavdata = {}
    txtdata = {}
    for index, filename in enumerate(glob.glob(os.path.join(complete_path_to_data_folder, '*.txt'))):
        file = np.loadtxt(filename)
        txtMIDI = convert_freq2midi(file[:, 2])
        thisfiledict = {"pitch": file[:, 2], "onset_seconds": file[:, 0], "freqInMIDI": txtMIDI}
        txtdata[index] = thisfiledict
    for index, filename in enumerate(glob.glob(os.path.join(complete_path_to_data_folder, '*.wav'))):
        [fs, data] = wf.read(filename)
        # Using annotation for variable block size.
        onsetInFlSamp = txtdata[index]['onset_seconds'] * fs
        onsetInSamp = np.zeros(len(onsetInFlSamp))
        for index1, item in enumerate(onsetInFlSamp):
            onsetInSamp[index1] = int(item)
        i = 1
        hopSize = np.zeros((len(onsetInSamp) - 1))
        while i < len(onsetInSamp):
            hopSize[i - 1] = np.floor(onsetInSamp[i] - onsetInSamp[i - 1])
            i += 1
        [dataf0, timeInSec] = track_pitch_acf(data, (hopSize * 2), hopSize, fs)
        freqInMIDI = convert_freq2midi(dataf0)
        thisfiledict = {"f0": dataf0, "timeInSec": timeInSec, "freqInMIDI": freqInMIDI}
        wavdata[index] = thisfiledict
    centsRMS = np.zeros(len(wavdata))
    for dict in wavdata:
        centsRMS[dict] = eval_pitchtrack(wavdata[dict]['f0'], txtdata[dict]['pitch'])
    return centsRMS


complete_path_to_data_folder = "homework1/trainData"
errorInCentsRMS = run_evaluation(complete_path_to_data_folder)
print(errorInCentsRMS)

# # Discussion
# Error for the three files are: 102.8, 477.6, 216.
# For get_f0_from_acf function, we improved upon scipy.signal.find_peaks. We only used primary and secondary peaks.
# We averaged the difference between adjacent samples.
# We also implemented a variable hopSize and blockSize based on the annotation data, which allowed us to more accurately
# compare time stamps across the estimate and ground truth.
# #

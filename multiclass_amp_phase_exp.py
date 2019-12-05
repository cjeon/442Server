import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from main import read_from_file
from main import fft_to_amplitude_band
from main import fft_to_phase_band
from itertools import chain

if __name__ == '__main__':
    # insert class name and file name here.
    classFileMap = {"heatgun": "heatgun.txt",
                    "box": "box.txt",
                    "glasses_box": "glasses_box.txt"}

    # import raw data
    rawDataMap = {}
    for className, fileName in classFileMap.items():
        rawDataMap[className] = read_from_file(fileName)

    # change to amplitude
    amplitudeMap = {}
    for className, rawData in rawDataMap.items():
       amplitudeMap[className] = list(map(fft_to_amplitude_band, rawData))

    # change to phase
    phaseMap = {}
    for className, rawData in rawDataMap.items():
       phaseMap[className] = list(map(fft_to_phase_band, rawData))

    # merge the amplitude and phase
    merged = {}
    for className in classFileMap.keys():
        # merged[className] = amplitudeMap[className] + phaseMap[className]
        # merged[className] = amplitudeMap[className]
        merged[className] = phaseMap[className]


    # make X and Y
    def flatmap(nested_list):
        return list(chain.from_iterable(nested_list))


    keys = merged.keys()
    X = flatmap(map(merged.get, keys))
    Y = flatmap( [key] * len(merged[key]) for key in keys)

    trainX, testX, trainY, testY = train_test_split(X, Y, shuffle=True, test_size=0.3)

    pca = PCA(10)
    trainX = pca.fit_transform(trainX)
    testX = pca.transform(testX)

    model = MLPClassifier(solver="lbfgs", max_iter=3000)
    model.fit(trainX, trainY)
    predY = model.predict(testX)

    print(classification_report(testY, predY))

    # model = GaussianNB()
    # model.fit(trainX, trainY)
    # predY = model.predict(testX)
    #
    # print(classification_report(testY, predY))